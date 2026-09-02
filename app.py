"""
Circuit Test Case Generator
----------------------------
A Streamlit app that takes an electronic/hardware circuit (as a schematic
image, a pasted netlist/description, or an uploaded file) and uses Gemini
to generate a structured set of test cases, which can then be downloaded
as a Word (.docx) document.

Run with:
    streamlit run app.py

Requires a GOOGLE_API_KEY (Gemini API key). You can either:
  - set it as an environment variable / in a .env file, or
  - paste it into the sidebar field when the app is running.

Additional dependency: this version generates test cases via a LangGraph
map-reduce pipeline (one focused LLM call per test-case category, then a
reduce step that merges + renumbers them), so you'll also need:
    pip install langgraph

Optional dependency: to see LangGraph's own raw/unedited ASCII export of
the compiled graph in the UI (proof it's a genuine LangGraph StateGraph,
alongside our expanded map-reduce diagram), also install:
    pip install grandalf
This is optional — the app works fine without it, it just skips that one
expander panel.
"""

import io
import json
import operator
import os
from datetime import datetime
from typing import Annotated, List, Optional, TypedDict

import streamlit as st
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, START, END

# Send() is what lets a LangGraph node fan out into several PARALLEL
# branches at runtime (the actual "map" in map-reduce). Its import path has
# moved between LangGraph versions, so fall back if the primary one isn't
# available in whatever version is installed.
try:
    from langgraph.types import Send
except ImportError:
    from langgraph.constants import Send


# --------------------------------------------------------------------------- #
# Config / constants
# --------------------------------------------------------------------------- #

load_dotenv()

GEMINI_MODEL_OPTIONS = [
    "gemini-3.5-flash-lite",
    "gemini-3.6-flash",
    "gemini-2.5-flash-lite",
    # "gemini-2.0-flash" removed — retired on Google's side (confirmed via
    # a live 404 NOT_FOUND from the API: "models/gemini-2.0-flash is no
    # longer available. Please update your code to use
    # models/gemini-3.6-flash for the latest features and improvements.")
]

st.set_page_config(
    page_title="Circuit Test Case Generator",
    page_icon="🔌",
    layout="wide",
)

# --------------------------------------------------------------------------- #
# Test-case categories — used by the LangGraph map-reduce pipeline below.
# Each category becomes one "map" call that focuses on just that slice of
# coverage, instead of one call trying to generate every kind of test case
# at once.
# --------------------------------------------------------------------------- #

CATEGORY_DEFINITIONS = {
    "functional": "Functional / nominal operating conditions",
    "boundary": "Boundary and edge cases (min/max voltage, current, frequency, timing, etc.)",
    "fault": "Fault / negative test cases (component failure, out-of-range input, short/open conditions)",
    "power": "Power-up / power-down / transient behavior",
    "protection": "Protection / safety-related behavior (fuses, clamps, current limiting, etc.)",
}


# --------------------------------------------------------------------------- #
# Data model for a single test case (used to force structured LLM output)
# --------------------------------------------------------------------------- #

class TestCase(BaseModel):
    test_id: str = Field(description="Short unique identifier, e.g. TC-001")
    title: str = Field(description="Short descriptive title of the test case")
    objective: str = Field(description="What this test case is verifying")
    detailed_explanation: str = Field(
        description=(
            "A thorough, technical explanation (4-6 sentences) of why this test case matters "
            "for THIS specific circuit — reference actual component values, signal names, or "
            "circuit behavior visible in the schematic/netlist. Explain what could go wrong if "
            "this behavior isn't verified, and how this test connects to real-world reliability "
            "or correctness of the circuit."
        )
    )
    preconditions: str = Field(description="Setup/preconditions required before running the test")
    test_steps: str = Field(
        description=(
            "Numbered steps to execute the test, formatted as a Markdown numbered list "
            "(e.g. '1. Apply 5V to VIN\\n2. Measure output at TP1\\n3. ...'). "
            "Each step on its own line."
        )
    )
    input_conditions: str = Field(description="Specific input values / stimulus applied to the circuit")
    expected_result: str = Field(description="Expected output / expected behavior of the circuit")
    pass_fail_criteria: str = Field(description="Objective criteria used to judge pass/fail")
    priority: str = Field(description="Priority: High, Medium, or Low")


class TestCaseCollection(BaseModel):
    circuit_summary: str = Field(description="Brief 2-4 sentence summary of the circuit under test")
    test_cases: List[TestCase] = Field(description="List of generated test cases")


class CircuitAnalysis(BaseModel):
    circuit_name: str = Field(description="Best-guess name/type of this circuit, e.g. 'Half Adder', 'RC Low-Pass Filter', 'Buck Converter'")
    inputs: List[str] = Field(description="List of identified input signal names, e.g. ['A', 'B']")
    outputs: List[str] = Field(description="List of identified output signal names, e.g. ['Sum', 'Carry']")
    description: str = Field(description="2-3 sentence plain-English explanation of what this circuit does")
    has_truth_table: bool = Field(description="True if this is a digital/logic circuit where a truth table makes sense; False for analog circuits")
    truth_table_markdown: str = Field(
        description=(
            "If has_truth_table is True, a complete Markdown table with one column per input/output "
            "and one row per input combination, showing the resulting output(s). "
            "If has_truth_table is False, return an empty string."
        )
    )
    recommended_test_case_count: int = Field(
        description=(
            "Your honest recommendation for how many distinct, meaningful test cases this circuit "
            "actually warrants (not padded with redundant cases). For a simple 2-input logic gate "
            "circuit this might be 4-8; for a more complex analog circuit it could be 10-20+."
        )
    )
    complexity_note: str = Field(
        description="One short sentence explaining why that many test cases are recommended (circuit complexity, number of I/O, etc.)"
    )
    applicable_categories: List[str] = Field(
        description=(
            "Which broad test-case categories genuinely apply to THIS circuit — a subset of "
            "['functional', 'boundary', 'fault', 'power', 'protection']. Always include "
            "'functional'. Include 'boundary' only if there are meaningful min/max or edge "
            "conditions. Include 'fault' only if failure modes are meaningful to test. Include "
            "'power' only if there's real power-up/down or transient behavior worth testing "
            "(e.g. skip it for a simple combinational logic gate). Include 'protection' only if "
            "the circuit visibly has protection elements (fuses, clamps, current limiting, etc.). "
            "Return only keys from this exact list — no other strings."
        )
    )


# --------------------------------------------------------------------------- #
# LLM helpers
# --------------------------------------------------------------------------- #

GENERATION_SYSTEM_PROMPT = """You are a senior hardware test/validation engineer.

You are given information about an electronic circuit (this may be a schematic
image, a netlist, or a text description). Your job is to analyze the circuit
and produce a thorough, professional set of test cases that a test engineer
could use to verify the circuit works as intended.

Cover, where relevant to the circuit shown:
- Functional / nominal operating conditions
- Boundary and edge cases (min/max voltage, current, frequency, timing, etc.)
- Fault / negative test cases (e.g. component failure, out-of-range input, short/open conditions)
- Power-up / power-down / transient behavior if applicable
- Any protection or safety-related behavior visible in the circuit (fuses, clamps, current limiting, etc.)

Write test cases that are specific to the actual circuit provided — do not
produce generic boilerplate. Reference actual component designators, nets,
or signal names from the circuit where they are visible/identifiable.

For every test case, write a genuinely thorough `detailed_explanation` —
do not write a one-line summary. Explain the engineering reasoning: what
this test verifies, why it matters for this specific circuit, what could
fail in the real world if this behavior is wrong, and how the test connects
to the circuit's overall correctness or reliability.

Do not generate more test cases than are genuinely meaningful for this
circuit's actual complexity — quality and depth of explanation matter more
than quantity. A simple 2-input logic circuit may only need 4-8 test cases;
do not pad the list with redundant or trivial variations just to reach a
higher count.

FORMATTING — do not use LaTeX or Markdown math syntax anywhere (no dollar
signs, no backslash commands like \\le, \\ge, \\text{...}, no subscript
braces like V_{IH}). This output is displayed as plain text and in a Word
document, neither of which renders LaTeX — it would show up as literal
raw syntax instead of a formatted equation. Instead, write thresholds and
formulas in plain, readable text using standard keyboard characters and
unicode symbols where helpful, for example:
  - "V_IH >= 2.0V" or "VIH ≥ 2.0 V" instead of "$V_{IH} \\ge 2.0\\text{V}$"
  - "R1 = 10k ohm" or "R1 = 10 kΩ" instead of "$R_1 = 10\\text{k}\\Omega$"
Plain component/signal names with underscores (V_IH, R_1) are fine; just
never wrap them in $ ... $ or use backslash-escaped LaTeX commands.
"""


def _strip_latex_artifacts(text: str) -> str:
    """Safety net in case the model still emits LaTeX/math syntax despite
    the system prompt instructing it not to. Converts common LaTeX patterns
    to plain, readable text so nothing shows up as raw '$...$' or '\\le'
    in the Streamlit display or the Word document, which don't render LaTeX."""
    if not text:
        return text
    import re

    s = text
    macro_map = {
        r"\\le": "≤", r"\\leq": "≤",
        r"\\ge": "≥", r"\\geq": "≥",
        r"\\times": "×", r"\\pm": "±",
        r"\\Omega": "Ω", r"\\mu": "μ",
        r"\\cdot": "·", r"\\%": "%",
        r"\\infty": "∞", r"\\approx": "≈",
    }
    for pattern, replacement in macro_map.items():
        s = re.sub(pattern, replacement, s)

    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)   # \text{V} -> V
    s = re.sub(r"_\{([^}]*)\}", r"_\1", s)       # V_{IH}   -> V_IH
    s = re.sub(r"\^\{([^}]*)\}", r"^\1", s)      # x^{2}    -> x^2
    s = s.replace("$", "")                        # strip leftover $ delimiters
    s = re.sub(r"\\[a-zA-Z]+", "", s)              # drop any remaining \command
    s = s.replace("{", "").replace("}", "")        # drop stray braces
    return s


def _clean_test_case_collection(collection: "TestCaseCollection") -> "TestCaseCollection":
    collection.circuit_summary = _strip_latex_artifacts(collection.circuit_summary)
    for tc in collection.test_cases:
        tc.title = _strip_latex_artifacts(tc.title)
        tc.objective = _strip_latex_artifacts(tc.objective)
        tc.detailed_explanation = _strip_latex_artifacts(tc.detailed_explanation)
        tc.preconditions = _strip_latex_artifacts(tc.preconditions)
        tc.test_steps = _strip_latex_artifacts(tc.test_steps)
        tc.input_conditions = _strip_latex_artifacts(tc.input_conditions)
        tc.expected_result = _strip_latex_artifacts(tc.expected_result)
        tc.pass_fail_criteria = _strip_latex_artifacts(tc.pass_fail_criteria)
    return collection


def _clean_circuit_analysis(analysis: "CircuitAnalysis") -> "CircuitAnalysis":
    analysis.description = _strip_latex_artifacts(analysis.description)
    analysis.complexity_note = _strip_latex_artifacts(analysis.complexity_note)
    analysis.truth_table_markdown = _strip_latex_artifacts(analysis.truth_table_markdown)
    return analysis


def get_model(api_key: str, model_name: str, temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=temperature,
    )


ANALYSIS_SYSTEM_PROMPT = """You are a senior hardware design engineer.

You are given information about an electronic circuit (a schematic image, a
netlist, and/or a text description). Identify what this circuit is, its
inputs and outputs, and explain briefly what it does.

If this is a digital/logic circuit (built from gates like AND, OR, XOR, NOT,
flip-flops, etc.), also produce a complete truth table as a Markdown table,
covering every combination of inputs.

If this is an analog circuit (filters, amplifiers, power supplies, etc.)
where a truth table does not apply, set has_truth_table to False and leave
truth_table_markdown as an empty string.

Also decide which test-case categories genuinely apply to this circuit —
see the `applicable_categories` field description for the exact rules and
allowed values. Be honest and selective here; a simple logic gate usually
doesn't need 'power' or 'protection' categories.

FORMATTING — do not use LaTeX or Markdown math syntax (no $ ... $, no
backslash commands like \\le, \\text{...}). Write any thresholds or
formulas in plain text with standard characters/unicode symbols instead
(e.g. "VIH >= 2.0V", not "$V_{IH} \\ge 2.0\\text{V}$").
"""


def _build_content_blocks(system_prompt: str, text_description: Optional[str],
                           image_bytes: Optional[bytes], image_mime_type: Optional[str]) -> list:
    content_blocks = [{"type": "text", "text": system_prompt}]

    if text_description:
        content_blocks.append({
            "type": "text",
            "text": f"\nCircuit description / netlist provided by the user:\n\n{text_description}",
        })

    if image_bytes:
        import base64
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        content_blocks.append({
            "type": "text",
            "text": "\nA schematic image of the circuit is attached below. Analyze it visually.",
        })
        content_blocks.append({
            "type": "image_url",
            "image_url": f"data:{image_mime_type};base64,{b64_image}",
        })

    return content_blocks


def analyze_circuit(
    api_key: str,
    model_name: str,
    text_description: Optional[str],
    image_bytes: Optional[bytes],
    image_mime_type: Optional[str],
) -> CircuitAnalysis:
    """Quick first pass: identify the circuit and produce a truth table if applicable."""
    model = get_model(api_key, model_name)
    structured_model = model.with_structured_output(CircuitAnalysis)

    content_blocks = _build_content_blocks(ANALYSIS_SYSTEM_PROMPT, text_description, image_bytes, image_mime_type)
    message = HumanMessage(content=content_blocks)
    analysis = structured_model.invoke([message])
    return _clean_circuit_analysis(analysis)


def _category_test_case_count(cats: List[str], total: int, idx: int) -> int:
    # Spread max_test_cases evenly across categories; give any remainder to
    # the first few categories so the total still adds up exactly to what
    # the user asked for.
    base = total // len(cats)
    remainder = total % len(cats)
    return base + (1 if idx < remainder else 0)


class GenerationGraphState(TypedDict):
    """State schema for the map-reduce test-case generation graph. Defined
    at module level (not nested inside a function) so it stays the single
    source of truth for the pipeline's real structure, used both for
    generation and to keep the UI's pipeline diagram consistent with it."""
    text_description: Optional[str]
    image_bytes: Optional[bytes]
    image_mime_type: Optional[str]
    api_key: str
    model_name: str
    max_test_cases: int
    categories: List[str]
    # Per-branch-only fields — each parallel invocation gets its own values
    # for these via Send(), so there's no shared-write conflict.
    category: str
    category_count: int
    # Annotated with operator.add so every parallel branch's returned list
    # gets CONCATENATED onto the running total. List concatenation is safe
    # with concurrent writes (order may vary, but reduce_node renumbers
    # everything afterward anyway, so order doesn't matter).
    test_cases: Annotated[List[TestCase], operator.add]
    # Set once, up front, before any parallel branch runs — never written
    # to by any map branch, so there's no concurrent-write conflict.
    circuit_summary: str
    # Separate, NON-accumulating key for the reduce step's finalized
    # output, so reduce_node's own write doesn't get concatenated onto the
    # already-accumulated `test_cases` (that was a real bug in an earlier
    # version of this pipeline — reduce must never write back to a field
    # with an operator.add reducer).
    final_test_cases: List[TestCase]


def _route_categories(state: GenerationGraphState) -> list:
    """This is the actual MAP fan-out: for each applicable category, spawn
    a parallel Send() to `generate_category`, each carrying its own copy of
    the shared state PLUS that category's name and count. LangGraph runs
    all of these branches concurrently."""
    sends = []
    for idx, cat in enumerate(state["categories"]):
        count = _category_test_case_count(state["categories"], state["max_test_cases"], idx)
        if count < 1:
            continue
        sends.append(Send("generate_category", {
            **state,
            "category": cat,
            "category_count": count,
        }))
    return sends


def _generate_category_node(state: GenerationGraphState) -> dict:
    category_key = state["category"]
    category_desc = CATEGORY_DEFINITIONS[category_key]
    count = state["category_count"]

    model = get_model(state["api_key"], state["model_name"])
    structured_model = model.with_structured_output(TestCaseCollection)

    system_prompt = (
        GENERATION_SYSTEM_PROMPT
        + f"\n\nFor THIS call, only generate test cases in the "
        + f"'{category_desc}' category — do not generate test cases "
        + f"for any other category. Generate EXACTLY {count} test case(s)."
    )

    content_blocks = _build_content_blocks(
        system_prompt, state["text_description"], state["image_bytes"], state["image_mime_type"]
    )
    message = HumanMessage(content=content_blocks)
    result = structured_model.invoke([message])
    result = _clean_test_case_collection(result)

    # Deliberately NOT returning circuit_summary here — see the
    # NOTE ON circuit_summary in generate_test_cases()'s docstring.
    return {"test_cases": result.test_cases}


def _reduce_node(state: GenerationGraphState) -> dict:
    # All parallel branches have finished by the time this node runs —
    # LangGraph doesn't advance past a normal (non-Send) edge until every
    # task from the fan-out has completed. Trim to the requested count and
    # renumber IDs so they're sequential/unique.
    cases = list(state["test_cases"][: state["max_test_cases"]])
    for i, tc in enumerate(cases, start=1):
        tc.test_id = f"TC-{i:03d}"
    return {"final_test_cases": cases}


def _build_generation_graph():
    """Builds and compiles the map-reduce StateGraph. Used for real
    generation (generate_test_cases); the pipeline diagram shown in the UI
    (render_pipeline_diagram) mirrors this same node structure (START ->
    per-category MAP -> REDUCE -> END) so it stays consistent with the
    graph that actually runs."""
    graph = StateGraph(GenerationGraphState)
    graph.add_node("generate_category", _generate_category_node)
    graph.add_node("reduce", _reduce_node)
    graph.add_conditional_edges(START, _route_categories, ["generate_category"])
    graph.add_edge("generate_category", "reduce")
    graph.add_edge("reduce", END)
    return graph.compile()


def get_pipeline_native_ascii() -> Optional[str]:
    """Returns LangGraph's OWN, unedited, auto-generated ASCII rendering of
    the real compiled graph (get_graph().draw_ascii()) — no custom drawing
    code of ours involved at all. This is here specifically as unedited
    proof that the pipeline is a genuine LangGraph StateGraph with real
    Send()-based map-reduce, for anyone (e.g. a reviewer/mentor) who wants
    to see LangGraph's own output rather than our expanded visualization.

    Because LangGraph can't statically know how many Send() calls a
    conditional-edge router will fire until the graph actually runs, this
    native export only shows ONE generic 'generate_category' box — not one
    per category — even though at runtime it fans out into N concurrent
    calls. That's expected and is exactly why render_pipeline_diagram()
    above draws an expanded version using the real category list: to make
    the actual runtime parallelism visible, since LangGraph's own static
    export can't show it.

    draw_ascii() (unlike draw_mermaid_png()) runs entirely locally with no
    network call, but does need the optional 'grandalf' package installed
    (pip install grandalf) — falls back to None if it's missing so the UI
    can degrade gracefully.
    """
    try:
        compiled_graph = _build_generation_graph()
        return compiled_graph.get_graph().draw_ascii()
    except Exception:
        return None


def _pipeline_svg(categories: List[str]) -> str:
    """Builds a self-contained inline SVG of the map-reduce pipeline.

    This is deliberately NOT a JS library (no mermaid.js/CDN) and NOT a
    remote-rendered image (no mermaid.ink) — it's plain SVG markup built as
    a Python string and handed straight to the browser, which renders SVG
    natively with zero extra network requests or scripts. That's what makes
    it reliable in locked-down/offline deployment environments, unlike the
    two previous approaches this app went through.

    The node names (START, one box per category, REDUCE, END) reflect the
    real compiled graph: LangGraph's Send() API fans a single
    'generate_category' node out into one concurrent call per category in
    `categories` (the actual list decided by the circuit analysis step),
    and 'reduce' only runs once every one of those calls has returned.
    """
    cats = categories or ["functional"]
    n = len(cats)

    box_w, box_h = 190, 58
    gap_x = 24
    row_gap = 70

    total_row_w = n * box_w + (n - 1) * gap_x
    canvas_w = max(total_row_w + 80, 420)
    canvas_h = 40 + 46 + row_gap + box_h + row_gap + box_h + row_gap + 46 + 30

    cx = canvas_w / 2
    start_y = 30
    start_h = 40
    row_y = start_y + start_h + row_gap
    reduce_y = row_y + box_h + row_gap
    end_y = reduce_y + box_h + row_gap
    end_h = 40

    row_left = cx - total_row_w / 2

    def esc(s: str) -> str:
        return (
            str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )

    parts = []
    parts.append(
        f'<svg viewBox="0 0 {canvas_w} {canvas_h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%; max-width:{int(canvas_w)}px; font-family:Calibri,Arial,sans-serif;">'
    )
    parts.append(
        '<defs><marker id="pipearrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#1F4E79" /></marker></defs>'
    )

    def rect(x, y, w, h, fill, stroke, label_lines, label_size=12, label_color="#1F2937", bold=False):
        r = [f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" ry="8" '
             f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />']
        n_lines = len(label_lines)
        line_h = label_size + 4
        start_ty = y + h / 2 - (n_lines - 1) * line_h / 2 + label_size / 3
        weight = "700" if bold else "500"
        for i, line in enumerate(label_lines):
            r.append(
                f'<text x="{x + w/2}" y="{start_ty + i*line_h}" text-anchor="middle" '
                f'font-size="{label_size}" font-weight="{weight}" fill="{label_color}">{esc(line)}</text>'
            )
        return "".join(r)

    def arrow(x1, y1, x2, y2):
        return (
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#1F4E79" stroke-width="1.75" marker-end="url(#pipearrow)" />'
        )

    # START
    start_x = cx - 60
    parts.append(rect(start_x, start_y, 120, start_h, "#1F4E79", "#1F4E79", ["START"], 13, "#FFFFFF", bold=True))

    # Fan-out arrows + category boxes (the real Send() map step)
    cat_centers = []
    for i, cat in enumerate(cats):
        bx = row_left + i * (box_w + gap_x)
        cat_centers.append(bx + box_w / 2)
        parts.append(arrow(cx, start_y + start_h, bx + box_w / 2, row_y))
        parts.append(
            rect(bx, row_y, box_w, box_h, "#EAF1F8", "#1F4E79",
                 ["MAP", f"generate_category", f"category = \"{cat}\""], 11)
        )

    # REDUCE
    reduce_x = cx - 100
    for bx_center in cat_centers:
        parts.append(arrow(bx_center, row_y + box_h, cx, reduce_y))
    parts.append(
        rect(reduce_x, reduce_y, 200, box_h, "#EAF1F8", "#1F4E79",
             ["REDUCE", "merge + trim + renumber"], 12)
    )

    # END
    end_x = cx - 60
    parts.append(arrow(cx, reduce_y + box_h, cx, end_y))
    parts.append(rect(end_x, end_y, 120, end_h, "#1F4E79", "#1F4E79", ["END"], 13, "#FFFFFF", bold=True))

    parts.append("</svg>")
    return "".join(parts)


def render_pipeline_diagram(categories: List[str]) -> None:
    """Renders the LangGraph pipeline as an inline SVG diagram in the
    Streamlit UI. Plain SVG, built locally from the graph's real
    node/category data — no JS library, no CDN, no external rendering
    service, so it can't fail to load the way a CDN script or a remote
    image API can in a restricted network environment."""
    try:
        svg = _pipeline_svg(categories)
        st.markdown(svg, unsafe_allow_html=True)
    except Exception:
        # Extremely defensive last resort — should not normally trigger,
        # since the SVG above has no external dependencies to fail.
        st.warning("Couldn't render the pipeline diagram — showing a text summary instead.")
        cats = categories or ["functional"]
        st.code(
            "START\n"
            + "\n".join(f"  -> MAP: generate_category(category=\"{c}\")" for c in cats)
            + "\n  -> REDUCE (merge + trim + renumber)\n  -> END",
            language=None,
        )


def _draw_pipeline_png(categories: List[str]) -> bytes:
    """Renders the same pipeline shape as _pipeline_svg(), but as a raster
    PNG via Pillow instead of SVG, so it can be embedded directly into the
    downloadable architecture .docx (python-docx needs raster image bytes,
    not SVG). Uses Pillow's built-in default font — no external font file
    or system dependency required, so this works the same in any
    environment Pillow already runs in (Pillow itself is already a
    transitive dependency of Streamlit, so nothing new needs installing).
    """
    cats = categories or ["functional"]
    n = len(cats)

    box_w, box_h = 220, 66
    gap_x = 26
    row_gap = 80
    margin = 40

    total_row_w = n * box_w + (n - 1) * gap_x
    canvas_w = max(total_row_w + margin * 2, 480)
    start_h, end_h = 46, 46
    canvas_h = margin + start_h + row_gap + box_h + row_gap + box_h + row_gap + end_h + margin

    img = Image.new("RGB", (int(canvas_w), int(canvas_h)), "white")
    draw = ImageDraw.Draw(img)

    navy = (31, 78, 121)
    light_fill = (234, 241, 248)
    white = (255, 255, 255)
    dark_text = (31, 41, 55)

    try:
        font_bold = ImageFont.truetype("DejaVuSans-Bold.ttf", 14)
        font_reg = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font_bold = ImageFont.load_default()
        font_reg = ImageFont.load_default()

    def centered_text(cx_, cy_, lines, font, fill):
        line_h = font.size + 6 if hasattr(font, "size") else 16
        total_h = line_h * len(lines)
        y = cy_ - total_h / 2 + line_h / 2
        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            w = bbox[2] - bbox[0]
            draw.text((cx_ - w / 2, y - (bbox[3] - bbox[1]) / 2 - bbox[1]), line, font=font, fill=fill)
            y += line_h

    def rounded_box(x, y, w, h, fill, outline):
        draw.rounded_rectangle([x, y, x + w, y + h], radius=10, fill=fill, outline=outline, width=2)

    def arrow(x1, y1, x2, y2):
        draw.line([x1, y1, x2, y2], fill=navy, width=2)
        # arrowhead
        import math
        angle = math.atan2(y2 - y1, x2 - x1)
        head_len = 10
        head_angle = math.radians(25)
        for sign in (-1, 1):
            hx = x2 - head_len * math.cos(angle + sign * head_angle)
            hy = y2 - head_len * math.sin(angle + sign * head_angle)
            draw.line([x2, y2, hx, hy], fill=navy, width=2)

    cx = canvas_w / 2
    start_y = margin
    row_y = start_y + start_h + row_gap
    reduce_y = row_y + box_h + row_gap
    end_y = reduce_y + box_h + row_gap
    row_left = cx - total_row_w / 2

    # START
    start_x = cx - 65
    rounded_box(start_x, start_y, 130, start_h, navy, navy)
    centered_text(cx, start_y + start_h / 2, ["START"], font_bold, white)

    # MAP boxes + fan-out arrows
    cat_centers = []
    for i, cat in enumerate(cats):
        bx = row_left + i * (box_w + gap_x)
        bcx = bx + box_w / 2
        cat_centers.append(bcx)
        arrow(cx, start_y + start_h, bcx, row_y)
        rounded_box(bx, row_y, box_w, box_h, light_fill, navy)
        centered_text(bcx, row_y + box_h / 2, ["MAP", "generate_category", f'category = "{cat}"'], font_reg, dark_text)

    # REDUCE + converging arrows
    reduce_x = cx - 110
    for bcx in cat_centers:
        arrow(bcx, row_y + box_h, cx, reduce_y)
    rounded_box(reduce_x, reduce_y, 220, box_h, light_fill, navy)
    centered_text(cx, reduce_y + box_h / 2, ["REDUCE", "merge + trim + renumber"], font_reg, dark_text)

    # END
    end_x = cx - 65
    arrow(cx, reduce_y + box_h, cx, end_y)
    rounded_box(end_x, end_y, 130, end_h, navy, navy)
    centered_text(cx, end_y + end_h / 2, ["END"], font_bold, white)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def get_pipeline_native_export():
    """Returns (text, kind) — LangGraph's own unedited export of the real
    compiled graph, preferring the ASCII form (needs the optional
    `grandalf` package) and falling back to the Mermaid text form (always
    available, no extra dependency) if that's not installed. Both are pure
    local text exports with no network call. Returns (None, None) only if
    LangGraph itself is somehow unable to export the graph at all."""
    try:
        compiled_graph = _build_generation_graph()
        graph = compiled_graph.get_graph()
    except Exception:
        return None, None

    try:
        return graph.draw_ascii(), "ascii"
    except Exception:
        pass
    try:
        return graph.draw_mermaid(), "mermaid"
    except Exception:
        return None, None


ARCH_DOC_ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ARCH_DOC_CODE_BG = "F2F2F2"


def _shade_paragraph(paragraph, fill_hex: str):
    p_pr = paragraph._p.get_or_add_pPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    p_pr.append(shd)


def _add_code_block(doc: Document, lines: List[str]):
    for line in lines:
        p = doc.add_paragraph()
        p.paragraph_format.space_before = Pt(0)
        p.paragraph_format.space_after = Pt(0)
        _shade_paragraph(p, ARCH_DOC_CODE_BG)
        run = p.add_run(line if line.strip() else " ")
        run.font.name = "Consolas"
        run.font.size = Pt(9)
    doc.add_paragraph()  # small gap after the block


def _add_body(doc: Document, text: str):
    p = doc.add_paragraph(text)
    p.paragraph_format.space_after = Pt(10)
    return p


def build_architecture_docx(
    illustrative_categories: List[str],
    actual_categories: List[str],
    native_export_text: Optional[str],
    native_export_kind: Optional[str],
) -> bytes:
    """Builds a structured, mentor-presentable Word document explaining
    and evidencing the LangGraph map-reduce pipeline: real code excerpts
    for each LangGraph primitive used (StateGraph, Send(), the
    operator.add reducer, the reduce node), the expanded pipeline diagram
    (both an illustrative multi-category run and the actual current run),
    and LangGraph's own unedited raw graph export as supporting proof."""
    doc = Document()

    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    title = doc.add_heading("Circuit Test Case Generator", level=0)
    for run in title.runs:
        run.font.color.rgb = ARCH_DOC_ACCENT
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run("LangGraph Map-Reduce Pipeline — Architecture Reference")
    sub_run.italic = True
    sub_run.font.size = Pt(12)
    doc.add_paragraph()

    # 1. Overview
    doc.add_heading("1. Overview", level=1)
    _add_body(
        doc,
        "Test cases are generated using a LangGraph StateGraph implementing the map-reduce "
        "pattern. The circuit-analysis step decides which coverage categories genuinely apply "
        "to a given circuit (functional, boundary, fault, power, protection). Each applicable "
        "category is then handled by its own concurrent LLM call (the \u201cmap\u201d step), and a "
        "single reduce step merges, trims, and renumbers all of their results once every call "
        "has completed (the \u201creduce\u201d step)."
    )

    # 2. Implementation evidence
    doc.add_heading("2. How Map-Reduce Is Implemented", level=1)
    _add_body(
        doc,
        "The pipeline uses LangGraph's real primitives directly \u2014 not a manual loop dressed "
        "up to look like map-reduce. The four building blocks below are all genuine LangGraph "
        "APIs, used as intended."
    )

    doc.add_heading("2.1  StateGraph \u2014 the graph itself", level=2)
    _add_code_block(doc, [
        "graph = StateGraph(GenerationGraphState)",
        'graph.add_node("generate_category", _generate_category_node)',
        'graph.add_node("reduce", _reduce_node)',
        'graph.add_conditional_edges(START, _route_categories, ["generate_category"])',
        'graph.add_edge("generate_category", "reduce")',
        'graph.add_edge("reduce", END)',
    ])
    _add_body(
        doc,
        "This is a real, compiled LangGraph StateGraph \u2014 the same object LangGraph uses "
        "internally for execution and for its own graph-export functions (Section 4)."
    )

    doc.add_heading('2.2  Send() \u2014 the real "map" fan-out', level=2)
    _add_code_block(doc, [
        "def _route_categories(state):",
        "    sends = []",
        '    for idx, cat in enumerate(state["categories"]):',
        "        count = _category_test_case_count(...)",
        '        sends.append(Send("generate_category", {**state, "category": cat}))',
        "    return sends",
    ])
    _add_body(
        doc,
        "Send() is LangGraph's own fan-out primitive. Returning a list of Send() objects tells "
        "LangGraph to invoke the \u201cgenerate_category\u201d node once per category, concurrently "
        "\u2014 this is the actual parallel \u201cmap\u201d step, scheduled by LangGraph itself, not by "
        "application code."
    )

    doc.add_heading("2.3  operator.add reducer \u2014 how parallel results are merged", level=2)
    _add_code_block(doc, [
        "class GenerationGraphState(TypedDict):",
        "    ...",
        "    test_cases: Annotated[List[TestCase], operator.add]",
    ])
    _add_body(
        doc,
        "Annotating a state field with operator.add tells LangGraph how to combine writes from "
        "multiple concurrent branches into one list. This reducer mechanism is what makes the "
        "pattern map-reduce rather than just \u201crun some things in parallel\u201d \u2014 LangGraph "
        "merges the branches' outputs itself, using this rule."
    )

    doc.add_heading("2.4  Reduce step \u2014 waits for every branch, then finalizes", level=2)
    _add_code_block(doc, [
        "def _reduce_node(state):",
        '    cases = list(state["test_cases"][: state["max_test_cases"]])',
        "    for i, tc in enumerate(cases, start=1):",
        '        tc.test_id = f"TC-{i:03d}"',
        '    return {"final_test_cases": cases}',
    ])
    _add_body(
        doc,
        "LangGraph does not run \u201creduce\u201d until every Send()-spawned branch has returned "
        "\u2014 this ordering guarantee comes from LangGraph's own scheduler, not from any manual "
        "wait/join logic in this code."
    )

    # 3. Diagrams
    doc.add_heading("3. Pipeline Diagram", level=1)
    _add_body(
        doc,
        "Diagrams below are generated directly from the real category list produced by the "
        "circuit-analysis step, to make the runtime fan-out visible."
    )

    doc.add_heading("3.1  Illustrative run \u2014 all applicable categories", level=2)
    illustrative_png = _draw_pipeline_png(illustrative_categories)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.add_run().add_picture(io.BytesIO(illustrative_png), width=Inches(6.0))
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_run = cap.add_run(
        "START fans out into one concurrent MAP call per category, then all branches converge "
        "into REDUCE, then END."
    )
    cap_run.italic = True
    cap_run.font.size = Pt(9)

    doc.add_heading("3.2  Actual current run", level=2)
    actual_png = _draw_pipeline_png(actual_categories)
    p2 = doc.add_paragraph()
    p2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    width_in = 3.0 if len(actual_categories or []) <= 1 else 6.0
    p2.add_run().add_picture(io.BytesIO(actual_png), width=Inches(width_in))
    cap2 = doc.add_paragraph()
    cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap2_run = cap2.add_run(
        "The categories shown here are exactly what the circuit-analysis step decided applied "
        "to this circuit."
    )
    cap2_run.italic = True
    cap2_run.font.size = Pt(9)

    # 4. Native export
    doc.add_heading("4. LangGraph's Own Raw Graph Export", level=1)
    if native_export_text:
        fn_name = "draw_ascii()" if native_export_kind == "ascii" else "draw_mermaid()"
        _add_body(
            doc,
            f"The export below is produced by calling compiled_graph.get_graph().{fn_name} \u2014 "
            "LangGraph's own built-in function, run against the exact compiled graph above, with "
            "no custom drawing code involved. It is included unedited as evidence that the "
            "pipeline is a genuine LangGraph StateGraph."
        )
        _add_code_block(doc, native_export_text.split("\n"))
        _add_body(
            doc,
            "Note: LangGraph's static export always shows a single generic \u201cgenerate_category\u201d "
            "box, because it cannot know in advance how many Send() calls a conditional router "
            "will produce \u2014 that is only decided at runtime. This is expected and is exactly "
            "why Section 3's diagrams exist \u2014 to make the real, per-run fan-out visible."
        )
    else:
        _add_body(
            doc,
            "LangGraph's raw graph export could not be generated in this environment (this "
            "does not affect the pipeline itself, only this proof panel)."
        )

    # 5. Summary
    doc.add_heading("5. Summary", level=1)
    for bullet in [
        "Uses a real, compiled LangGraph StateGraph (Section 2.1).",
        "Map step uses LangGraph's own Send() fan-out primitive (Section 2.2).",
        "Reduce step uses LangGraph's operator.add reducer to merge concurrent branch outputs (Section 2.3).",
        "LangGraph's own scheduler \u2014 not manual code \u2014 guarantees reduce waits for every branch (Section 2.4).",
        "Confirmed against LangGraph's own unedited graph export (Section 4).",
    ]:
        doc.add_paragraph(bullet, style="List Bullet")

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


def generate_test_cases(
    api_key: str,
    model_name: str,
    text_description: Optional[str],
    image_bytes: Optional[bytes],
    image_mime_type: Optional[str],
    max_test_cases: int = 15,
    categories: Optional[List[str]] = None,
    circuit_summary_hint: Optional[str] = None,
) -> TestCaseCollection:
    """Generates test cases via a LangGraph map-reduce pipeline — PARALLEL
    version.

    MAP step: one focused LLM call PER CATEGORY (functional, boundary,
    fault, power, protection — whichever apply), fired off concurrently via
    LangGraph's Send() API instead of looping through them one at a time.
    Each call only has to think about one slice of coverage.

    REDUCE step: once every parallel category call has finished, one final
    node merges all their results, trims to `max_test_cases`, and
    renumbers everything as TC-001, TC-002, ... (each category's own call
    numbers its own test cases starting from TC-001, so renumbering is
    what prevents duplicate/colliding IDs across categories).

    NOTE ON PARALLELISM: running categories concurrently means multiple
    requests hit the Gemini API at (roughly) the same time. This is more
    likely to trip a free-tier rate limit than the old sequential version —
    if you see rate-limit errors, switching models in the sidebar (each has
    its own separate quota) is the workaround.

    NOTE ON circuit_summary: each category's own structured output includes
    a `circuit_summary` field, but we deliberately IGNORE it here and don't
    let any map branch write to a shared `circuit_summary` state key.
    Reason: with true parallel branches, if two categories finished at the
    same time and both tried to write to the same state key with no merge
    rule, LangGraph would raise a conflict error — it doesn't know how to
    combine two simultaneous writes to a plain (non-accumulating) field.
    Instead, the summary is carried straight through from the circuit
    analysis step (`circuit_summary_hint`), which only ever gets set once,
    up front, before any parallel branch runs.

    The graph structure itself lives in `_build_generation_graph()` (module
    level); the UI's pipeline diagram (see `render_pipeline_diagram()`)
    mirrors this same START -> per-category MAP -> REDUCE -> END shape.
    """

    if not categories:
        categories = list(CATEGORY_DEFINITIONS.keys())
    categories = [c for c in categories if c in CATEGORY_DEFINITIONS] or ["functional"]

    compiled_graph = _build_generation_graph()

    initial_state: GenerationGraphState = {
        "text_description": text_description,
        "image_bytes": image_bytes,
        "image_mime_type": image_mime_type,
        "api_key": api_key,
        "model_name": model_name,
        "max_test_cases": max_test_cases,
        "categories": categories,
        "category": "",
        "category_count": 0,
        "test_cases": [],
        "circuit_summary": circuit_summary_hint or "",
        "final_test_cases": [],
    }

    final_state = compiled_graph.invoke(initial_state)

    return TestCaseCollection(
        circuit_summary=final_state["circuit_summary"] or "Circuit summary unavailable.",
        test_cases=final_state["final_test_cases"],
    )


# --------------------------------------------------------------------------- #
# Word document generation
# --------------------------------------------------------------------------- #

ACCENT_COLOR = RGBColor(0x1F, 0x4E, 0x79)


def _set_cell_shading(cell, fill_hex: str):
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill_hex)
    tc_pr.append(shd)


def _set_table_fixed_layout(table):
    """Force Word to respect explicit column widths instead of auto-sizing
    columns based on cell content. Without this, table.columns[i].width is
    treated as a hint only, and Word will happily widen a column past the
    page margin to fit long text — which is why a column like 'Pass/Fail'
    can get pushed off the visible page even though widths were "set"."""
    tbl_pr = table._tbl.tblPr
    layout = OxmlElement("w:tblLayout")
    layout.set(qn("w:type"), "fixed")
    tbl_pr.append(layout)


def _apply_column_widths(table, widths):
    """Set the width on every cell in every row (not just table.columns).
    python-docx's table.columns[i].width only updates the shared grid
    definition; individual cells created via add_row() keep their own width
    unless it's set explicitly here, and Word falls back to autofit if any
    cell disagrees with the grid."""
    for row in table.rows:
        for cell, w in zip(row.cells, widths):
            cell.width = w


def build_docx(circuit_name: str, collection: TestCaseCollection) -> bytes:
    doc = Document()

    # Give the tables a bit more breathing room than Word's 1" default
    # margins so the fixed column widths below comfortably fit within the
    # printable page area on both letter and A4 paper.
    for section in doc.sections:
        section.left_margin = Inches(0.75)
        section.right_margin = Inches(0.75)
        section.top_margin = Inches(0.75)
        section.bottom_margin = Inches(0.75)

    # Base font
    style = doc.styles["Normal"]
    style.font.name = "Calibri"
    style.font.size = Pt(10.5)

    # Title
    title = doc.add_heading("Circuit Test Case Report", level=0)
    for run in title.runs:
        run.font.color.rgb = ACCENT_COLOR
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER

    subtitle = doc.add_paragraph()
    subtitle.alignment = WD_ALIGN_PARAGRAPH.CENTER
    sub_run = subtitle.add_run(circuit_name or "Untitled Circuit")
    sub_run.italic = True
    sub_run.font.size = Pt(12)

    meta = doc.add_paragraph()
    meta.alignment = WD_ALIGN_PARAGRAPH.CENTER
    meta_run = meta.add_run(f"Generated on {datetime.now().strftime('%B %d, %Y')}")
    meta_run.font.size = Pt(9)
    meta_run.font.color.rgb = RGBColor(0x59, 0x59, 0x59)

    doc.add_paragraph()

    # Circuit summary
    doc.add_heading("Circuit Summary", level=1)
    doc.add_paragraph(collection.circuit_summary)

    doc.add_paragraph()

    # --- At-a-glance summary table ---
    doc.add_heading("Test Case Summary (At a Glance)", level=1)
    summary_table = doc.add_table(rows=1, cols=5)
    summary_table.style = "Table Grid"
    summary_table.alignment = WD_TABLE_ALIGNMENT.LEFT
    summary_table.autofit = False
    _set_table_fixed_layout(summary_table)
    # Widths sum to 6.5" — fits inside the 7.0" printable width (8.5" letter
    # width minus 0.75" margins on each side) with room to spare.
    col_widths = [Inches(0.55), Inches(1.75), Inches(0.7), Inches(2.6), Inches(0.9)]
    for col, w in zip(summary_table.columns, col_widths):
        col.width = w

    header_cells = summary_table.rows[0].cells
    headers = ["ID", "Title", "Priority", "Expected Result", "Pass/Fail"]
    for cell, header_text in zip(header_cells, headers):
        cell.text = ""
        p = cell.paragraphs[0]
        r = p.add_run(header_text)
        r.bold = True
        r.font.size = Pt(10)
        r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
        _set_cell_shading(cell, "1F4E79")

    for tc in collection.test_cases:
        row_cells = summary_table.add_row().cells
        values = [tc.test_id, tc.title, tc.priority, tc.expected_result, ""]
        for cell, val in zip(row_cells, values):
            cell.text = ""
            p = cell.paragraphs[0]
            r = p.add_run(str(val))
            r.font.size = Pt(9.5)

        # Light shading for priority cell based on level, so it's scannable at a glance
        priority_colors = {"high": "FCE4E4", "medium": "FFF3CD", "low": "E2F0D9"}
        _set_cell_shading(row_cells[2], priority_colors.get(tc.priority.strip().lower(), "FFFFFF"))

    # Re-apply widths now that all rows exist — add_row() cells otherwise
    # default back to an even split of the table width.
    _apply_column_widths(summary_table, col_widths)

    doc.add_paragraph()
    doc.add_page_break()
    doc.add_heading(f"Detailed Test Cases ({len(collection.test_cases)})", level=1)

    # One table per test case for readability
    for tc in collection.test_cases:
        header_para = doc.add_paragraph()
        header_run = header_para.add_run(f"{tc.test_id} — {tc.title}")
        header_run.bold = True
        header_run.font.size = Pt(12)
        header_run.font.color.rgb = ACCENT_COLOR

        table = doc.add_table(rows=0, cols=2)
        table.alignment = WD_TABLE_ALIGNMENT.LEFT
        table.style = "Table Grid"
        table.autofit = False
        _set_table_fixed_layout(table)
        # Widths sum to 6.5" — fits inside the 7.0" printable width.
        detail_col_widths = [Inches(1.6), Inches(4.9)]
        table.columns[0].width = detail_col_widths[0]
        table.columns[1].width = detail_col_widths[1]

        rows_data = [
            ("Objective", tc.objective),
            ("Priority", tc.priority),
            ("Why This Test Matters", tc.detailed_explanation),
            ("Preconditions", tc.preconditions),
            ("Test Steps", tc.test_steps),
            ("Input Conditions", tc.input_conditions),
            ("Expected Result", tc.expected_result),
            ("Pass/Fail Criteria", tc.pass_fail_criteria),
            ("Actual Result", ""),  # left blank for the tester to fill in during execution
            ("Pass / Fail", ""),    # left blank for the tester to fill in during execution
            ("Tested By / Date", ""),  # left blank for the tester to fill in during execution
        ]

        for label, value in rows_data:
            row_cells = table.add_row().cells
            row_cells[0].text = ""
            label_para = row_cells[0].paragraphs[0]
            label_run = label_para.add_run(label)
            label_run.bold = True
            label_run.font.size = Pt(10)
            _set_cell_shading(row_cells[0], "F2F2F2")

            row_cells[1].text = ""
            value_para = row_cells[1].paragraphs[0]
            if value:
                value_run = value_para.add_run(str(value))
                value_run.font.size = Pt(10)
            else:
                # Blank fill-in field (Actual Result / Pass-Fail / Tested By)
                # — add a faint placeholder and extra blank lines for writing in.
                placeholder_run = value_para.add_run("[to be filled in during testing]")
                placeholder_run.font.size = Pt(9)
                placeholder_run.italic = True
                placeholder_run.font.color.rgb = RGBColor(0xAA, 0xAA, 0xAA)
                row_cells[1].add_paragraph()  # extra blank line for handwritten/typed entry

        # Re-apply widths now that all rows exist — same reason as the
        # summary table above.
        _apply_column_widths(table, detail_col_widths)

        doc.add_paragraph()

    buf = io.BytesIO()
    doc.save(buf)
    buf.seek(0)
    return buf.getvalue()


# --------------------------------------------------------------------------- #
# Streamlit UI
# --------------------------------------------------------------------------- #

def main():
    # Top header with Clear button
    title_col, clear_col = st.columns([6, 1])

    with title_col:
        st.title("🔌 Circuit Test Case Generator")

    with clear_col:
        st.write("")
        clear_clicked = st.button("🗑️ Clear", use_container_width=True)

    if clear_clicked:
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_model", None)
        st.session_state.pop("collection", None)
        st.session_state.pop("collection_model", None)
        st.session_state.pop("circuit_name", None)
        st.session_state.pop("circuit_name_input", None)
        st.session_state.pop("pasted_text", None)
        st.session_state.pop("image_uploader", None)
        st.session_state.pop("file_uploader", None)
        st.rerun()

    st.caption(
        "Upload or describe a circuit → generate structured test cases → "
        "download as a Word document."
    )

    with st.sidebar:
        st.header("Settings")

        api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            api_key = st.text_input(
                "Gemini API key",
                type="password",
                help="Paste your GOOGLE_API_KEY here, or set it as an environment variable / in a .env file instead.",
            )

        model_name = st.selectbox("Gemini model", GEMINI_MODEL_OPTIONS, index=0)
        st.caption(f"✅ Currently using: **{model_name}**")

        st.markdown("---")
        st.markdown(
            "**Note:** If you hit a rate-limit error, try switching to a "
            "different model above — each model has its own separate quota. "
            "The model shown in your results confirms which one actually ran."
        )

        st.markdown("---")
        st.header("📜 History (this session)")

        history = st.session_state.get("history", [])

        if not history:
            st.caption("Nothing generated yet. Results will appear here after you click 'Generate Test Cases'.")
        else:
            st.caption(
                f"{len(history)} generation(s) so far. Note: history clears if you refresh or close this tab "
                "— it isn't saved permanently, so download anything you want to keep."
            )
            for idx, entry in enumerate(reversed(history)):
                real_idx = len(history) - 1 - idx
                with st.expander(f"{entry['timestamp']} — {entry['circuit_name']} ({entry['num_test_cases']} TCs)"):
                    st.caption(f"Model: {entry['model']}")
                    st.download_button(
                        label="⬇️ Download this Word doc",
                        data=entry["docx_bytes"],
                        file_name=f"{entry['circuit_name'].replace(' ', '_')}_test_cases.docx",
                        mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        key=f"history_download_{real_idx}",
                        use_container_width=True,
                    )
                    if st.button("↩️ Reload this result", key=f"history_reload_{real_idx}", use_container_width=True):
                        st.session_state["collection"] = entry["collection"]
                        st.session_state["collection_model"] = entry["model"]
                        st.session_state["circuit_name"] = entry["circuit_name"]
                        st.rerun()

            if st.button("🗑️ Clear history", use_container_width=True):
                st.session_state["history"] = []
                st.rerun()

    circuit_name = st.text_input("Circuit name / label",placeholder="e.g. Buck Converter v2, Power Supply Rev A",key="circuit_name_input",)

    tab_image, tab_netlist, tab_file = st.tabs(["📷 Upload schematic image", "📝 Paste netlist / description", "📁 Upload file"])

    image_bytes = None
    image_mime_type = None
    text_description = None

    with tab_image:
        uploaded_image = st.file_uploader(
            "Upload a schematic image (PNG, JPG)",
            type=["png", "jpg", "jpeg"],
            key="image_uploader",
        )
        if uploaded_image is not None:
            image_bytes = uploaded_image.read()
            image_mime_type = uploaded_image.type
            preview_col, _ = st.columns([1, 2])
            with preview_col:
                st.image(image_bytes, caption="Uploaded schematic", width=350)

    with tab_netlist:
        pasted_text = st.text_area(
            "Paste a netlist, circuit description, or component list",
            key="pasted_text",
            height=250,
            placeholder=(
                "Example:\n"
                "R1 (10k) between VIN and node A\n"
                "C1 (100nF) between node A and GND\n"
                "U1 op-amp, non-inverting configuration...\n\n"
                "Or a SPICE-style netlist, or a plain-English description of the circuit."
            ),
        )
        if pasted_text and pasted_text.strip():
            text_description = pasted_text.strip()

    with tab_file:
        uploaded_file = st.file_uploader(
            "Upload a text-based circuit file (.txt, .net, .cir, .md)",
            type=["txt", "net", "cir", "md"],
            key="file_uploader",
        )
        if uploaded_file is not None:
            file_text = uploaded_file.read().decode("utf-8", errors="ignore")
            st.text_area("File contents preview", value=file_text, height=200, disabled=True)
            text_description = (text_description or "") + "\n\n" + file_text

    st.markdown("---")

    has_input = bool(image_bytes) or bool(text_description)

    analyze_clicked = st.button(
        "🔍 Analyze Circuit (name + truth table)",
        use_container_width=True,
        disabled=not has_input,
    )

    if analyze_clicked:
        if not api_key:
            st.error("No Gemini API key found. Add GOOGLE_API_KEY to your .env file, or paste one into the sidebar field.")
            st.stop()

        with st.spinner(f"Identifying circuit using {model_name}..."):
            try:
                analysis = analyze_circuit(
                    api_key=api_key,
                    model_name=model_name,
                    text_description=text_description,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                )
                st.session_state["analysis"] = analysis
                st.session_state["analysis_model"] = model_name
            except Exception as e:
                st.error(f"Analysis failed: {e}")
                st.stop()

    if "analysis" in st.session_state:
        analysis: CircuitAnalysis = st.session_state["analysis"]

        st.subheader(f"🔎 {analysis.circuit_name}")
        st.caption(f"Generated using: {st.session_state.get('analysis_model', 'unknown model')}")
        st.write(analysis.description)

        col_in, col_out = st.columns(2)
        with col_in:
            st.markdown(f"**Inputs:** {', '.join(analysis.inputs) if analysis.inputs else '—'}")
        with col_out:
            st.markdown(f"**Outputs:** {', '.join(analysis.outputs) if analysis.outputs else '—'}")

        if analysis.has_truth_table and analysis.truth_table_markdown:
            st.markdown("**Truth Table**")
            st.markdown(analysis.truth_table_markdown)
        else:
            st.info("This appears to be an analog circuit — a truth table doesn't apply here.")

        st.markdown(
            f"📊 **Recommended test cases for this circuit: ~{analysis.recommended_test_case_count}**  \n"
            f"_{analysis.complexity_note}_"
        )

        category_labels = [
            CATEGORY_DEFINITIONS.get(c, c) for c in analysis.applicable_categories
        ]
        st.caption(
            "🗂️ Map-reduce categories for generation: " + ", ".join(category_labels)
        )

        with st.expander("🔀 View the LangGraph map-reduce pipeline"):
            st.caption(
                "This is the actual graph structure used to generate test cases below — "
                "one focused LLM call PER CATEGORY, fired in PARALLEL (MAP, via LangGraph's "
                "Send() fan-out), then merged and renumbered once all of them finish (REDUCE)."
            )
            render_pipeline_diagram(analysis.applicable_categories)
            cats = analysis.applicable_categories
            st.caption(
                f"⚡ {len(cats)} categor{'y' if len(cats) == 1 else 'ies'} running in parallel this "
                "run: " + ", ".join(cats) + ". Parallel calls are faster but more likely to hit "
                "free-tier rate limits than running them one at a time — switch models in the "
                "sidebar if you see a rate-limit error."
            )

    st.markdown("---")

    # Number of test cases to generate — placed here (after truth table / recommendation),
    # capped to what's actually meaningful once a circuit has been analyzed.
    if "analysis" in st.session_state:
        recommended = st.session_state["analysis"].recommended_test_case_count
        options_max = max(recommended, 1)
        default_value = recommended
        help_text = (
            f"Options capped to {options_max} based on the analyzed circuit's complexity — "
            f"the model recommended ~{recommended} meaningful test cases for this circuit."
        )
    else:
        options_max = 25
        default_value = 12
        help_text = "Click 'Analyze Circuit' above first to get a tailored, circuit-specific option list."

    test_case_options = list(range(1, options_max + 1))
    default_index = test_case_options.index(default_value) if default_value in test_case_options else len(test_case_options) - 1

    max_test_cases = st.selectbox(
        "How many test cases should be generated?",
        options=test_case_options,
        index=default_index,
        help=help_text,
    )

    if "analysis" not in st.session_state:
        st.caption("💡 Analyze a circuit first to tailor this list automatically.")

    st.markdown("---")

    generate_col, clear_col = st.columns([4, 1])

    with generate_col:
        generate_clicked = st.button(
            "🧪 Generate Test Cases",
            type="primary",
            use_container_width=True,
            disabled=not has_input,
        )

    with clear_col:
        clear_clicked = st.button(
            "🗑️ Clear",
            use_container_width=True,
            key="generate_clear_button",
        )

    if clear_clicked:
        st.session_state.pop("analysis", None)
        st.session_state.pop("analysis_model", None)
        st.session_state.pop("collection", None)
        st.session_state.pop("collection_model", None)
        st.session_state.pop("circuit_name", None)
        st.session_state.pop("circuit_name_input", None)
        st.session_state.pop("pasted_text", None)
        st.session_state.pop("image_uploader", None)
        st.session_state.pop("file_uploader", None)
        st.rerun()

    if generate_clicked:
        if not api_key:
            st.error("No Gemini API key found. Add GOOGLE_API_KEY to your .env file, or paste one into the sidebar field.")
            st.stop()

        if not has_input:
            st.error("Please provide at least one input: a schematic image, a pasted description, or an uploaded file.")
            st.stop()

        with st.spinner(f"Analyzing circuit and generating test cases using {model_name} (map-reduce across categories, running in parallel)..."):
            try:
                # If the circuit was already analyzed, use the categories the
                # model decided actually apply (functional/boundary/fault/
                # power/protection). Otherwise fall back to all categories —
                # the map-reduce pipeline in generate_test_cases() will
                # default to that itself, but being explicit here keeps the
                # applicable-categories decision visible at the call site.
                applicable_categories = (
                    st.session_state["analysis"].applicable_categories
                    if "analysis" in st.session_state
                    else None
                )
                # Carry the circuit summary through from the analysis step
                # instead of asking each parallel category call to generate
                # its own — see the circuit_summary note inside
                # generate_test_cases() for why that matters with true
                # parallel (Send-based) branches.
                circuit_summary_hint = (
                    st.session_state["analysis"].description
                    if "analysis" in st.session_state
                    else None
                )
                collection = generate_test_cases(
                    api_key=api_key,
                    model_name=model_name,
                    text_description=text_description,
                    image_bytes=image_bytes,
                    image_mime_type=image_mime_type,
                    max_test_cases=max_test_cases,
                    categories=applicable_categories,
                    circuit_summary_hint=circuit_summary_hint,
                )
                resolved_name = circuit_name or (
                    st.session_state["analysis"].circuit_name if "analysis" in st.session_state else ""
                )
                st.session_state["collection"] = collection
                st.session_state["collection_model"] = model_name
                st.session_state["circuit_name"] = resolved_name

                # Save to history so past results aren't lost when a new circuit is analyzed
                if "history" not in st.session_state:
                    st.session_state["history"] = []
                st.session_state["history"].append({
                    "timestamp": datetime.now().strftime("%H:%M:%S"),
                    "circuit_name": resolved_name or "Untitled Circuit",
                    "model": model_name,
                    "num_test_cases": len(collection.test_cases),
                    "collection": collection,
                    "docx_bytes": build_docx(resolved_name, collection),
                })
            except Exception as e:
                st.error(f"Generation failed: {e}")
                st.stop()

        # Rerun so the sidebar (rendered earlier in the script) picks up
        # the freshly-saved history entry on this next pass, instead of
        # showing last run's (stale) history.
        st.rerun()

    # Display results if we have them (persisted across reruns via session_state)
    if "collection" in st.session_state:
        collection: TestCaseCollection = st.session_state["collection"]

        st.success(f"Generated {len(collection.test_cases)} test cases using **{st.session_state.get('collection_model', 'unknown model')}**.")

        st.subheader("Circuit Summary")
        st.write(collection.circuit_summary)

        st.subheader("Test Case Summary")

        # Build a Markdown table instead of st.dataframe — Streamlit's dataframe
        # component truncates long cell text with no way to wrap it, even in
        # fullscreen. A Markdown table wraps naturally like normal page text.
        summary_md_lines = [
            "| ID | Title | Priority | Expected Result |",
            "|---|---|---|---|",
        ]
        for tc in collection.test_cases:
            # Escape pipe characters so they don't break the table structure
            title = tc.title.replace("|", "\\|")
            expected = tc.expected_result.replace("|", "\\|")
            summary_md_lines.append(f"| {tc.test_id} | {title} | {tc.priority} | {expected} |")

        st.markdown("\n".join(summary_md_lines))

        st.subheader("Full Test Case Details")
        for tc in collection.test_cases:
            with st.expander(f"{tc.test_id} — {tc.title}  (Priority: {tc.priority})"):
                detail_rows = {
                    "Objective": tc.objective,
                    "Why This Test Matters": tc.detailed_explanation,
                    "Preconditions": tc.preconditions,
                    "Test Steps": tc.test_steps,
                    "Input Conditions": tc.input_conditions,
                    "Expected Result": tc.expected_result,
                    "Pass/Fail Criteria": tc.pass_fail_criteria,
                }

                import html as _html

                table_rows_html = ""
                for label, value in detail_rows.items():
                    # Convert simple numbered-list text (e.g. "1. ...\n2. ...")
                    # into real <br> line breaks so it renders as intended
                    # inside the HTML table cell instead of one run-on line.
                    escaped_value = _html.escape(str(value)).replace("\n", "<br>")
                    escaped_label = _html.escape(label)
                    # NOTE: built as a single line with no leading indentation —
                    # Markdown treats any line indented 4+ spaces as a code block,
                    # which was causing the raw HTML tags to display as literal text
                    # instead of rendering. Keeping every line unindented fixes this.
                    table_rows_html += (
                        f'<tr>'
                        f'<td style="background-color:#F2F2F2; font-weight:600; padding:10px 12px; width:180px; vertical-align:top; border:1px solid #DDD;">{escaped_label}</td>'
                        f'<td style="padding:10px 12px; vertical-align:top; border:1px solid #DDD;">{escaped_value}</td>'
                        f'</tr>'
                    )

                full_table_html = f'<table style="width:100%; border-collapse:collapse; margin-bottom:12px;">{table_rows_html}</table>'
                st.markdown(full_table_html, unsafe_allow_html=True)

        docx_bytes = build_docx(st.session_state.get("circuit_name", ""), collection)

        st.download_button(
            label="⬇️ Download Test Cases as Word Document",
            data=docx_bytes,
            file_name=f"{(circuit_name or 'circuit').replace(' ', '_')}_test_cases.docx",
            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            type="primary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()
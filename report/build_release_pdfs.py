"""Build the three-page technical report and one-page AI-use disclosure."""
from __future__ import annotations

import json
import re
import statistics
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (BaseDocTemplate, Frame, PageBreak, PageTemplate,
                                Paragraph, Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report"
NAVY = colors.HexColor("#16324F")
TEAL = colors.HexColor("#147D86")
PALE = colors.HexColor("#EAF1F5")
INK = colors.HexColor("#202B33")
MUTED = colors.HexColor("#586875")
WHITE = colors.white


def read_json(path: str):
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def build_styles():
    return {
        "title": ParagraphStyle("TitleCustom", fontName="Helvetica-Bold", fontSize=17,
                                leading=20, textColor=NAVY, spaceAfter=2),
        "subtitle": ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=8.1,
                                   leading=10, textColor=MUTED, spaceAfter=7),
        "h": ParagraphStyle("HeadingCustom", fontName="Helvetica-Bold", fontSize=10.2,
                            leading=12, textColor=TEAL, spaceBefore=4, spaceAfter=3),
        "body": ParagraphStyle("BodyCustom", fontName="Helvetica", fontSize=8.15,
                               leading=10.15, textColor=INK, spaceAfter=3),
        "small": ParagraphStyle("Small", fontName="Helvetica", fontSize=7.3,
                                leading=8.6, textColor=INK, spaceAfter=2),
        "tiny": ParagraphStyle("Tiny", fontName="Helvetica", fontSize=6.9,
                               leading=8, textColor=INK),
        "note": ParagraphStyle("Note", fontName="Helvetica-Oblique", fontSize=7.4,
                               leading=9, textColor=MUTED, spaceBefore=2, spaceAfter=3),
        "disclosure": ParagraphStyle("Disclosure", fontName="Helvetica", fontSize=8.1,
                                      leading=10, textColor=INK, spaceAfter=4),
    }


def para(text: str, style):
    return Paragraph(text, style)


def make_table(data, widths, font=7.5, header=True, aligns=None, padd=3):
    table = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    cmds = [
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font),
        ("TEXTCOLOR", (0, 0), (-1, -1), INK),
        ("LEADING", (0, 0), (-1, -1), font + 1.3),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), padd),
        ("RIGHTPADDING", (0, 0), (-1, -1), padd),
        ("TOPPADDING", (0, 0), (-1, -1), padd),
        ("BOTTOMPADDING", (0, 0), (-1, -1), padd),
        ("LINEBELOW", (0, -1), (-1, -1), 0.35, colors.HexColor("#CCD6DE")),
    ]
    if header:
        cmds += [("BACKGROUND", (0, 0), (-1, 0), NAVY),
                 ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                 ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                 ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY)]
        for row in range(1, len(data)):
            if row % 2 == 0:
                cmds.append(("BACKGROUND", (0, row), (-1, row), PALE))
    if aligns:
        for col, align in enumerate(aligns):
            cmds.append(("ALIGN", (col, 0), (col, -1), align))
    table.setStyle(TableStyle(cmds))
    return table


def footer(canvas, doc):
    canvas.saveState()
    w, _ = letter
    canvas.setStrokeColor(colors.HexColor("#D6E0E7"))
    canvas.setLineWidth(0.5)
    canvas.line(36, 29, w - 36, 29)
    canvas.setFont("Helvetica", 7)
    canvas.setFillColor(MUTED)
    canvas.drawString(36, 17, "QC@IITI Qiskit Fall Fest 2026 | Quantum Traffic Gridlock")
    canvas.drawRightString(w - 36, 17, f"Page {doc.page}")
    canvas.restoreState()


def get_block3_limitation():
    text = (ROOT / "docs" / "checkpoint3_regret.md").read_text(encoding="utf-8")
    row = next(line for line in text.splitlines() if line.startswith("| candidate S/2 |"))
    fields = [part.strip() for part in row.strip("|").split("|")]
    worst = re.search(r"([0-9.]+)%", fields[4]).group(1)
    r_regret = re.search(r"([0-9.]+)%", fields[9]).group(1)
    elite = fields[11]
    return worst, r_regret, elite


def build_technical_report():
    b6 = read_json("outputs/R/block6_metrics.json")
    baselines = read_json("outputs/R/baselines.json")
    ablation = read_json("outputs/block7/ablation_results.json")
    generalization = read_json("outputs/block7/generalization.json")
    depth_text = (ROOT / "docs" / "qaoa_ablation.md").read_text(encoding="utf-8", errors="replace")
    derivation = (ROOT / "docs" / "derivation.md").read_text(encoding="utf-8")
    ai_log = (ROOT / "AI_USAGE_LOG.md").read_text(encoding="utf-8")
    assert "Neither R nor H was used" in derivation
    assert "Neither R nor H" in depth_text
    assert "10.42%" in ai_log and "47.11%" in ai_log

    styles = build_styles()
    output = OUT / "report.pdf"
    doc = BaseDocTemplate(str(output), pagesize=letter, leftMargin=36, rightMargin=36,
                          topMargin=35, bottomMargin=37, title="Quantum Traffic Gridlock - Technical Report",
                          author="QC@IITI Qiskit Fall Fest 2026")
    frame = Frame(36, 37, letter[0] - 72, letter[1] - 72, id="normal")
    doc.addPageTemplates([PageTemplate(id="all", frames=frame, onPage=footer)])
    story = []

    # Page 1: model, QUBO, and constant selection.
    story += [para("Quantum Traffic Gridlock", styles["title"]),
              para("Technical report | QC@IITI Qiskit Fall Fest 2026 | Scenario-independent QUBO and QAOA", styles["subtitle"]),
              para("1. Traffic model and notation", styles["h"]),
              para("Two intersections (A and B) are scheduled over four slots. Let x[A,t] and x[B,t] be binary: 1 selects north-south (NS), and 0 selects east-west (EW). The canonical eight-bit order is A1,A2,A3,A4,B1,B2,B3,B4. The official scorer adds arrivals and prior-slot transfers, serves up to capacity C on the green approach, and scores the remaining queues. Its transfer is floor(f * departures + 0.5), delivered to B at the start of the next slot; transfers after slot 4 leave the network. Official cost is delay + w_ovf * overflow + w_sw * switches.", styles["body"]),
              para("2. Scenario-independent QUBO", styles["h"]),
              para("For each approach-slot, the builder forms a linear queue proxy q_i(x)=c_i+a_i^T x. It accumulates external arrivals, approximates green service as C*x for NS or C*(1-x) for EW, and models A-to-B arrivals from the next slot onward as f*C times the linked-phase green indicator. Unlike the official model, these proxy terms do not clip service to available queue and do not round fractional transfers. The scorer remains authoritative for schedule costs.", styles["body"]),
              para("The surrogate is C_tilde(x)=sum_i q_i(x) + gamma*sum_i(q_i(x)-alpha*S)^2 + w_sw*sum(x_t XOR x_(t+1)), with alpha=0.50. Delay has weight 1; switching uses the scenario's w_sw and the exact binary identity x XOR y=x+y-2xy. The quadratic congestion term approximates the scorer's max(0, queue-S) overflow hinge. For q=c+a^T x, expansion uses x_i^2=x_i: the constant is c_shift^2, diagonal terms are (2*c_shift*a_i+a_i^2), and pair terms are 2*a_i*a_j, multiplied by gamma. No auxiliary variables are used; there are eight QUBO bits.", styles["body"]),
              para("The penalty scale is gamma=2*w_ovf/S^2. The multiplier 2 was retained from the first 150 scenarios of the 200-scenario random bank generated with seed 20261015 (arrivals 0-8, capacity 4-6, storage 5-8, overflow weight 2-6, switch weight 0.25-1.5, link fraction 0.4-0.8). The center alpha was then compared at 0.25, 0.50, and 1.00 on a separate 100-scenario bank generated with seed 77441931; alpha=0.50 won by mean regret, P90, worst regret, then correlation. R and H were excluded from both tuning banks. The seed-77441931 set was used for candidate selection and is described as a holdout in the source report.", styles["body"]),
              para("Limit of the surrogate: linear service can make proxy queues negative, fractional transfers differ from half-up integers, and a smooth square is not the scorer's clipped overflow hinge.", styles["note"]),
              PageBreak()]

    # Page 2: QAOA, comparison, and five-seed metrics.
    qaoa = b6["per_seed"]
    comp = b6["comparison"]
    raw_row = next(row for row in comp if row["method"].startswith("QAOA raw"))
    seed_one = next(row for row in qaoa if row["seed"] == 1)
    hybrid_best = b6["hybrid_best"]
    main_schedule = read_json("outputs/R/schedule.json")
    depth_rows = re.findall(
        r"^\|\s*(1|2)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|\s*([0-9.]+)\s*\|",
        depth_text, re.MULTILINE)
    depth = {int(p): {"gap": float(gap), "time": float(seconds)}
             for p, gap, _median, seconds in depth_rows}
    evaluation_summary = ", ".join(
        f"{row['optimizer_evaluations']} (seed {row['seed']})" for row in qaoa)
    comp_data = [["Method", "Schedule / interpretation", "Official cost"]]
    for row in comp:
        sched = row["schedule"]
        if row["method"].startswith("QAOA raw"):
            sched = "5 x 4,096 raw shots; E_Q mean"
        comp_data.append([row["method"], sched, f"{row['official_cost']:.6f}"])
    seed_data = [["Seed", "E_Q", "Lift", "Elite enrichment"]]
    for row in qaoa:
        seed_data.append([str(row["seed"]), f"{row['E_Q']:.9f}", f"{row['lift']:.9f}", f"{row['elite_enrichment']:.9f}"])
    min_lift = b6["minimum_lift_clipped"]
    story += [para("3. QAOA design and R benchmark", styles["h"]),
              para(f"With z_i=1-2*x_i, the upper-triangular QUBO becomes E(z)=K+sum_i h_i*z_i+sum_(i&lt;j) J_ij*z_i*z_j. Here K=offset+0.5*sum_i Q_ii+0.25*sum_(i&lt;j)Q_ij, h_i=-Q_ii/2-0.25*sum_(j&#8800;i)Q_ij, and J_ij=Q_ij/4. Exhaustive parity over the 256 assignments has maximum error {main_schedule['qaoa']['ising_max_assignment_error']:.3g} for R. Each layer starts with Hadamards, applies RZ/RZZ cost rotations and RX mixers.", styles["body"]),
              para(f"Production uses p=1, COBYLA, a seeded linear-ramp initialization with uniform jitter +/-0.01, and a hard 300-objective-evaluation cap. The Ising cost coefficients are divided by their uniform-sampling standard deviation before constructing the cost rotations; the QUBO and official scores are unchanged. The depth ablation used 16 random scenarios (seed 20261016, excluding R/H): normalized gaps were {depth[1]['gap']:.5f} for p=1 and {depth[2]['gap']:.5f} for p=2, with mean optimizer times {depth[1]['time']:.4f} s and {depth[2]['time']:.4f} s. We fixed p=1 to reduce layer and runtime cost. R's optimizer evaluations by seed were {evaluation_summary}. Final sampling uses exactly 4,096 shots and the run seed. Qiskit prints c7..c0; decoding reverses that string to A1..A4,B1..B4.", styles["body"]),
              para("R method comparison (C*=72.000; C_rand=151.250)", styles["h"]),
              make_table(comp_data, [185, 225, 94], font=7.25, aligns=["LEFT", "LEFT", "RIGHT"], padd=2.6),
              para(f"The raw QAOA row is the official-cost mean over all {len(qaoa) * 4096:,} raw shots: {raw_row['official_cost']:.9f}. Seed 1's most-frequent raw schedule is {seed_one['raw_most_frequent_bits']} (cost {seed_one['raw_most_frequent_official_cost']:.1f}). The separately labelled hybrid row selects the best official-cost schedule present in sampled bitstrings across the five runs; it found {hybrid_best['bits']} at {hybrid_best['official_cost']:.1f}. This postselection is not the raw-QAOA mean.", styles["note"]),
              para("Per-seed raw-QAOA metrics", styles["h"]),
              make_table(seed_data, [50, 155, 150, 150], font=7.3, aligns=["CENTER", "RIGHT", "RIGHT", "RIGHT"], padd=2.5),
              para(f"Minimum lift clipped to [0,1] over the five seeds: {min_lift:.12f}. Elite enrichment is the fraction of raw shots in the 13 lowest-cost schedules (ties by ascending bitstring), divided by 13/256. H was not present in the repository.", styles["note"]),
              PageBreak()]

    # Page 3: ablations, generalization, limitations.
    factor_groups = [("Depth", "p=" + str(p), "p", p) for p in (1,2,3,4)]
    factor_groups += [("Gamma", "x" + f"{s:g}", "penalty_scale", s) for s in (0.5,1.0,2.0)]
    factor_groups += [("Init", name, "initialization", name) for name in ("linear-ramp","random","fixed")]
    factor_data = [["Factor", "Level", "Expected official cost", "Lift", "Elite enrichment"]]
    for label, level, key, value in factor_groups:
        subset = [r for r in ablation["runs"] if r[key] == value]
        factor_data.append([label, level,
                            f"{statistics.mean(r['expected_official_cost'] for r in subset):.3f}",
                            f"{statistics.mean(r['lift'] for r in subset):.5f}",
                            f"{statistics.mean(r['elite_enrichment'] for r in subset):.4f}"])
    gen_data = [["Scenario", "Mean E_Q", "Mean lift", "Min clipped lift", "Mean enrichment", "Max evals", "Max QAOA s"]]
    for row in generalization:
        gen_data.append([row["scenario"], f"{row['mean_E_Q']:.6f}", f"{row['mean_lift']:.6f}",
                         f"{row['minimum_lift_clipped']:.6f}", f"{row['mean_elite_enrichment']:.4f}",
                         str(row["max_evaluations"]), f"{row['max_runtime_per_seed_seconds']:.3f}"])
    worst, rreg, relite = get_block3_limitation()
    r_generalization = next(row for row in generalization if row["scenario"] == "R")
    story += [para("4. Ablations, generalization, and limitations", styles["h"]),
              para("The full ablation covers p=1..4, gamma multipliers x0.5/x1/x2, and linear-ramp/random/fixed initialization, crossed over five seeds on R plus four fresh hidden-range scenarios (bank seed 20261017). There were 900 configurations; each had an 80-evaluation cap. Values below average all scenarios, seeds, and settings for the other factors. Ideal-statevector expectations are used for this ablation; the separate generalization table uses the normal 4,096-shot CLI.", styles["body"]),
              make_table(factor_data, [48, 65, 142, 104, 146], font=7.0, padd=2.3),
              para("Unmodified default pipeline on R and the four random scenarios, five seeds each", styles["h"]),
              make_table(gen_data, [54, 86, 68, 90, 93, 55, 67], font=6.9, padd=2.1),
              para(f"Honest limitations. On R, the exact QUBO minimizer is 11101001 with official cost 79.5, 10.42% regret relative to C*=72. Its overlap with the official elite set is 4/13. In the selected-center 100-scenario bank, the worst exact-QUBO regret is 47.11%; this is a heavy tail despite the lower mean. Raw QAOA R mean lift is {r_generalization['mean_lift']:.6f} and its minimum clipped lift is {r_generalization['minimum_lift_clipped']:.6f}. The hybrid happened to find the exact optimum in these samples, but raw sampling is evaluated separately from the raw mean. CVaR was not implemented. Scenario H was unavailable. Model mismatch sources include unclipped linear service, fractional unrounded transfers, and the smooth overflow penalty.", styles["body"]),
              para(f"Source table details: candidate S/2 R regret {rreg} and R elite overlap {relite}; worst holdout regret {worst}. Generalization seed totals and all per-seed rows are in outputs/block7/generalization.json.", styles["note"])]
    doc.build(story)
    return output


def build_ai_disclosure():
    styles = build_styles()
    output = OUT / "ai_disclosure.pdf"
    doc = BaseDocTemplate(str(output), pagesize=letter, leftMargin=42, rightMargin=42,
                          topMargin=42, bottomMargin=40, title="AI-use disclosure",
                          author="QC@IITI Qiskit Fall Fest 2026")
    frame = Frame(42, 40, letter[0]-84, letter[1]-82, id="disclosure")
    doc.addPageTemplates([PageTemplate(id="all", frames=frame, onPage=footer)])
    story = [para("AI-use disclosure", styles["title"]),
             para("QC@IITI Qiskit Fall Fest 2026 | Quantum Traffic Gridlock", styles["subtitle"]),
             para("AI tools and records", styles["h"]),
             para("The existing AI_USAGE_LOG.md records AI-assisted formulation, traffic-model and baseline code, metrics, QUBO derivation, calibration scripts, and report tables for Blocks 1-3. It does not name the AI product or model used in those blocks. Codex assisted with the Block 5-8 implementation and release drafting in this task thread. Python, Qiskit, Qiskit Aer, SciPy, pytest, ReportLab, and Poppler were used for execution, testing, PDF generation, and visual checks; these are software dependencies/tools, not claims about the historical AI model.", styles["disclosure"]),
             para("AI-generated work", styles["h"]),
             para("AI assistance produced formulation ideas, model and baseline code, QUBO construction and derivation text, calibration and metric scripts, the QAOA circuit/optimizer and artifact writer, Block 6 and Block 7 analysis scripts, automated tests, and the current report/disclosure/README drafts. The original AI_USAGE_LOG.md stops at Block 3; the later work is evidenced in the current source tree and task outputs, not in that original log.", styles["disclosure"]),
             para("Independent checks recorded in AI_USAGE_LOG.md", styles["h"]),
             para("The log records the author's check of the four official R vectors (116.0, 146.0, 237.0, 92.5), review of scenario parameters against the problem statement, review of baseline and elite-set logic, and checking the derivation equations against src/qubo.py. It does not record the author's independent sign-off on all-256 scorer parity, QUBO/Ising parity, or circuit/count consistency. The repository contains automated tests for those checks; an automated pass is not represented as a personal verification.", styles["disclosure"]),
             para("AUTHOR CONFIRMATION REQUIRED BEFORE SUBMISSION", styles["h"]),
             para("Leave each box unchecked until you have personally confirmed the statement:", styles["small"])]
    confirmations = [
        "[ ] I independently confirm scorer.py parity on all 256 schedules. (The test suite compares R and 100 generated scenarios; the current AI log records the four published vectors only.)",
        "[ ] I independently confirm QUBO-to-Ising energy equality, including sign and offset, for all 256 assignments.",
        "[ ] I independently confirm the saved bound circuit, parameters, and 4,096-shot counts are consistent and reproducible.",
        "[ ] I independently confirm I have read docs/derivation.md and understand its approximations and limits. (The AI log records an equation review.)",
    ]
    for statement in confirmations:
        story += [para(statement, styles["disclosure"]),
                  para("Author initials: ____________________    Date: ______________", styles["small"]),
                  Spacer(1, 4)]
    story += [para("Historical AI product/model names are missing from AI_USAGE_LOG.md. Add them if known; do not infer them from this release task.", styles["note"])]
    doc.build(story)
    return output


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    print(build_technical_report())
    print(build_ai_disclosure())





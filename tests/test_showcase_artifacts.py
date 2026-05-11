from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPACE_DIR = ROOT / "huggingface-space"


def test_huggingface_poster_contains_required_story_sections() -> None:
    html = (SPACE_DIR / "index.html").read_text(encoding="utf-8")

    required_sections = [
        "Positioning",
        "Take-Home Scope",
        "Motivation and Fit",
        "Deeter Analytics",
        "Amazon Nova",
        "End-to-End System",
        "Success Criteria",
        "System Architecture",
        "pipeline-figure.svg",
        "https://github.com/duoan/mega-trading",
        "github.com/duoan/mega-trading",
        "https://github.com/duoan/mega-trading/blob/main/huggingface-space/technical-report.pdf",
        "technical-report.pdf",
        "technical-report.tex",
        "Model Architecture",
        "Training System",
        "Ablation Study",
        "Hypotheses",
        "ablation-figure.svg",
        "Engineering Design Evidence",
        "https://github.com/duoan/mega-trading/blob/main/docs/kernels.md",
        "docs/kernels.md",
        "docs/performance-profiling.md",
        "156.23",
        "323.645 ms",
        "Optimization Stack",
        "Inference Path",
        "Future Direction",
        "Reference Frame",
        "Multimodal",
        "L3 Data",
    ]
    for section in required_sections:
        assert section in html


def test_huggingface_poster_embeds_local_rtx_metrics() -> None:
    html = (SPACE_DIR / "index.html").read_text(encoding="utf-8")

    expected_metrics = [
        "2.3379",
        "10.3592",
        "49.23%",
        "69.42%",
        "54,695,345",
        "6,201,267",
        "108K",
        "Baseline Sanity Check",
        "0.17%",
        "33.63%",
        "31.91%",
    ]
    for metric in expected_metrics:
        assert metric in html


def test_latex_technical_report_has_methods_and_limitations() -> None:
    report = (SPACE_DIR / "technical-report.tex").read_text(encoding="utf-8")

    required_phrases = [
        "\\documentclass",
        "\\begin{abstract}",
        "Mega-Trading",
        "pipeline-figure.pdf",
        "\\includegraphics[width=\\linewidth]{pipeline-figure.pdf}",
        "discrete event token",
        "decoder-only Transformer",
        "\\bibitem{tradefm}",
        "\\bibitem{llama3}",
        "\\bibitem{triton}",
        "chronological backtest",
        "Take-Home Scope",
        "Motivation and Fit",
        "Deeter Analytics",
        "Amazon Nova",
        "research engineer",
        "Ablation Study",
        "Hypotheses",
        "ablation-figure.pdf",
        "Engineering Design Evidence",
        "https://github.com/duoan/mega-trading/blob/main/docs/kernels.md",
        "docs/kernels.md",
        "docs/performance-profiling.md",
        "156.23",
        "323.645 ms",
        "price-depth L1",
        "end-to-end system",
        "success criteria",
        "version control",
        "closed-loop\nmarket simulator",
        "Accuracy Baselines",
        "Multimodal",
        "L3",
    ]
    for phrase in required_phrases:
        assert phrase in report


def test_pipeline_figure_is_available_as_standalone_svg() -> None:
    svg = (SPACE_DIR / "pipeline-figure.svg").read_text(encoding="utf-8")

    required_labels = [
        "Mega-Trading architecture figure",
        "DATA PLANE",
        "TOKENIZATION",
        "MODEL CORE",
        "TRAINING AND EVALUATION",
        "INFERENCE OUTPUTS",
    ]
    for label in required_labels:
        assert label in svg


def test_pipeline_figure_is_available_as_embedded_pdf_source() -> None:
    figure_source = (SPACE_DIR / "pipeline-figure.tex").read_text(encoding="utf-8")

    assert (SPACE_DIR / "pipeline-figure.pdf").is_file()
    required_labels = [
        "Mega-Trading Architecture",
        "DATA PLANE",
        "TOKENIZATION",
        "MODEL CORE",
        "TRAINING",
        "EVALUATION",
        "kernel: Triton GQA forward",
        "156.23 TFLOP/s",
        "profiling: optimizer CPU window",
        "323.645 ms to 20.018 ms",
    ]
    for label in required_labels:
        assert label in figure_source


def test_ablation_figure_is_available_as_standalone_svg() -> None:
    svg = (SPACE_DIR / "ablation-figure.svg").read_text(encoding="utf-8")

    required_labels = [
        "Ablation study",
        "Backtest loss",
        "Price-depth L1",
        "Top-1 accuracy",
        "small_100",
    ]
    for label in required_labels:
        assert label in svg

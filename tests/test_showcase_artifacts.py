from pathlib import Path
import re


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
        "IEX DEEP/HIST",
        "https://iextrading.com/trading/market-data/",
    ]
    for section in required_sections:
        assert section in html


def test_project_readme_exposes_reviewer_links() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_links = [
        "open trading foundation model",
        "public-data baseline",
        "git clone https://github.com/duoan/mega-trading.git",
        "curl -LsSf https://astral.sh/uv/install.sh | sh",
        "uv --version",
        "cd mega-trading",
        "uv sync",
        "make demo",
        "mac",
        "local smoke run",
        "rtx",
        "RTX PRO 6000 Blackwell",
        "single-server GPU baseline",
        "modal",
        "cloud GPU path",
        "## Key Links",
        "https://huggingface.co/spaces/duoan/mega-trading",
        "https://github.com/duoan/mega-trading/blob/main/huggingface-space/technical-report.pdf",
        "huggingface-space/index.html",
        "huggingface-space/backtest-report-rtx.html",
        "huggingface-space/pipeline-figure.pdf",
        "docs/ablation-results.md",
        "docs/performance-profiling.md",
        "docs/kernels.md",
    ]
    for link in required_links:
        assert link in readme


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


def test_huggingface_poster_opens_external_links_in_new_tabs() -> None:
    html = (SPACE_DIR / "index.html").read_text(encoding="utf-8")
    external_links = re.findall(r'<a\b(?=[^>]*href="https?://)[^>]*>', html)

    assert external_links
    for link in external_links:
        assert 'target="_blank"' in link
        assert 'rel="noopener noreferrer"' in link


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
        "\\bibitem{iexmarketdata}",
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
        "IEX DEEP/HIST",
        "https://iextrading.com/trading/market-data/",
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


def test_huggingface_space_deploy_workflow_is_configured() -> None:
    workflow = ROOT / ".github" / "workflows" / "deploy-huggingface-space.yml"
    workflow_text = workflow.read_text(encoding="utf-8")
    space_readme = (SPACE_DIR / "README.md").read_text(encoding="utf-8")

    required_workflow_phrases = [
        "Deploy Hugging Face Space",
        "duoan/mega-trading",
        "huggingface-space/**",
        "workflow_dispatch",
        "secrets.HF_TOKEN",
        "hf upload",
        "--repo-type space",
        "technical-report.pdf",
    ]
    for phrase in required_workflow_phrases:
        assert phrase in workflow_text
    assert "huggingface-cli upload" not in workflow_text

    required_space_metadata = [
        "sdk: static",
        "app_file: index.html",
    ]
    for phrase in required_space_metadata:
        assert phrase in space_readme

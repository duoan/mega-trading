# Mega-Trading Static Space

This folder is ready to upload to a Hugging Face Static Space.

- `index.html`: poster-style project page.
- `pipeline-figure.svg`: standalone architecture figure used by the overview page.
- `pipeline-figure.tex`: TikZ source for the paper-style architecture figure embedded in the technical report.
- `pipeline-figure.pdf`: compiled architecture figure embedded in the technical report.
- `ablation-figure.svg`: standalone ablation chart used by the overview page.
- `ablation-figure.pdf`: ablation chart embedded by the LaTeX report.
- `technical-report.pdf`: compiled technical report. The poster links to the GitHub PDF at https://github.com/duoan/mega-trading/blob/main/huggingface-space/technical-report.pdf.
- `technical-report.tex`: primary LaTeX technical report source.
- `technical-report.md`: Markdown backup for quick browser reading.
- `backtest-report-rtx.html`: copied local deep-dive backtest dashboard for the `rtx` run.

Reviewer-facing engineering references live one directory up under `docs/`, especially [`docs/kernels.md`](https://github.com/duoan/mega-trading/blob/main/docs/kernels.md), [`docs/performance-profiling.md`](https://github.com/duoan/mega-trading/blob/main/docs/performance-profiling.md), [`docs/data-plane-design.md`](https://github.com/duoan/mega-trading/blob/main/docs/data-plane-design.md), [`docs/training-plane-design.md`](https://github.com/duoan/mega-trading/blob/main/docs/training-plane-design.md), [`docs/training-configs.md`](https://github.com/duoan/mega-trading/blob/main/docs/training-configs.md), and [`docs/training-systems-alignment.md`](https://github.com/duoan/mega-trading/blob/main/docs/training-systems-alignment.md).

The files were generated from local `.mega-trading/data` artifacts. Re-run the project pipeline and regenerate this folder when metrics change.

# Contributing

Contributions from biologists, network scientists, software engineers, and trainees are welcome.

## Before proposing a method

Please describe the scientific question, required inputs, assumptions, expected output, and at least one methodological reference. New rankings must clearly distinguish structural association or model prediction from causal evidence.

## Local development

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
streamlit run app.py
```

Run checks before opening a pull request:

```bash
ruff check .
pytest
```

Never contribute identifiable human-subject data. Use synthetic, public, or properly governed de-identified examples.

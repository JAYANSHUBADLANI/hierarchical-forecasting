PY := ./.venv/bin/python

.PHONY: demo data study figures test clean

demo: data test study figures
	@echo
	@echo "Tables in outputs/tables, figures in outputs/figures."

data:
	$(PY) scripts/fetch_data.py

study:
	$(PY) -u scripts/run_study.py

figures:
	$(PY) scripts/make_figures.py

test:
	$(PY) -m pytest

clean:
	rm -rf outputs/tables/* outputs/figures/* .pytest_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

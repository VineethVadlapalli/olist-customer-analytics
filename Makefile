.PHONY: setup data features notebooks app test all

setup:      ## install Python 3.12 env + dependencies (needs uv)
	uv sync

data:       ## download the raw Olist CSVs into data/raw
	uv run python scripts/download_data.py

features: data  ## build customer / cohort / model tables in DuckDB + app parquet files
	uv run python src/features.py
	uv run python src/export_app_data.py

notebooks:  ## re-run both analysis notebooks (outputs + charts saved)
	cd notebooks && for nb in 01_cohorts_ltv_segments 02_repeat_purchase_model; do \
		uv run jupytext --to ipynb $$nb.py -q && uv run jupyter nbconvert --to notebook --execute --inplace $$nb.ipynb; done

app:        ## start the dashboard at http://localhost:8501
	uv run streamlit run app/streamlit_app.py

test:       ## smoke-test the dashboard
	uv run pytest -q tests

all: features notebooks test

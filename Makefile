CONFIG ?= configs/release.yaml
UV ?= uv

.PHONY: setup lock lint typecheck test smoke reproduce gate docker app clean

setup:            ## Create the environment from the lockfile
	$(UV) sync --frozen --all-extras
	$(UV) run pre-commit install

lock:             ## Re-resolve dependencies (commit uv.lock afterwards)
	$(UV) lock

lint:
	$(UV) run ruff check src tests app
	$(UV) run ruff format --check src tests app

typecheck:
	$(UV) run mypy

test:
	$(UV) run pytest --cov --cov-report=term-missing --cov-report=xml

smoke:            ## Full pipeline on synthetic data (~1 min)
	$(UV) run parpi reproduce -c configs/smoke.yaml

reproduce:        ## Full pipeline on the real GEO data
	$(UV) run parpi reproduce -c $(CONFIG)

gate:
	$(UV) run parpi gate -c $(CONFIG)

docker:
	docker build -t parpi-resistance:dev .

app:              ## Run the web app against the locally trained model
	cd app && PARPI_MODEL=../outputs/models/$$(cat ../outputs/models/SELECTED) $(UV) run python app.py

clean:
	rm -rf .smoke .pytest_cache .mypy_cache .ruff_cache coverage.xml

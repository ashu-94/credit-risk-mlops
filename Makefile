.PHONY: data train eda explain threshold serve demo test drift docker all

data:
	python data/generate_data.py --n 20000 --out data/credit.csv

train:
	python -m src.train

benchmark:
	python -m src.benchmark

eda:
	python -m src.eda

explain:
	python -m src.explain

threshold:
	python -m src.threshold

serve:
	uvicorn app.main:app --reload

demo:
	streamlit run app/streamlit_app.py

test:
	python -m pytest tests/ -q

drift:
	python data/generate_data.py --n 3000 --out data/new_batch.csv --seed 7
	python -m src.monitor --current data/new_batch.csv

# Full offline run: data -> train -> eda -> explain -> threshold
all: data train eda explain threshold

docker:
	docker compose up --build

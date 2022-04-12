init:
	pip install -r requirements.txt
	pip install -e .
	chmod +x scripts/download_data.sh

download_data:
	./scripts/download_data.sh
.PHONY: help
help: ## Show this help
	@egrep -h '\s##\s' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install:  ## Install Visio
	pip install protobuf==3.20
	pip install git+https://github.com/huggingface/transformers.git@c612628045822f909020f7eb6784c79700813eda
	cd src/GPTQ-for-LLaMa && git checkout cuda && pip install -r requirements.txt
	cd src/GPTQ-for-LLaMa && python setup_cuda.py install
conda create --name "visio-gptq" python=3.10 -y
conda activate "visio-gptq"
pip install protobuf==3.20
pip install git+https://github.com/huggingface/transformers.git@c612628045822f909020f7eb6784c79700813eda
cd GPTQ-for-LLaMa && pip install -r requirements.txt && python setup_cuda.py install
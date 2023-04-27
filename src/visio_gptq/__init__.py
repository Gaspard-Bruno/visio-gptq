import sys
from pathlib import Path
current_path = Path(__file__)
current_dir = current_path.parent
sys.path.insert(0, str(Path(f"{current_dir}/GPTQ-for-LLaMa")))

from llama_inference import load_quant # type: ignore

sys.path.insert(0, str(Path(f"{current_path}")))
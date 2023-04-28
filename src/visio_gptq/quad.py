import os

from pathlib import Path
import visio_gptq.download_model as downloader
from visio_gptq import load_quant # type: ignore
CURRENT_PATH = Path(__file__)


def load_quantized(model_name, wbits=4, groupsize=128, device="cuda"):
    if not os.path.exists(model_name):
        new_model_name = model_name.replace('/', '_')
        path_to_model = Path(f'{CURRENT_PATH.parent}/models/{new_model_name}')
        if not os.path.exists(path_to_model):
            model, branch = downloader.sanitize_model_and_branch_names(model_name, 'main')
            links, sha256, _ = downloader.get_download_links_from_huggingface(model, branch)
            os.mkdir(path_to_model)
            downloader.download_model_files(model,branch=branch, links=links, sha256=sha256, output_folder=path_to_model)
    else:
        path_to_model = Path(model_name)
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) >= 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) >= 1:
        pt_path = found_safetensors[0]

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

    model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, device=device)

    return model

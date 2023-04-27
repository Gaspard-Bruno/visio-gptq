from transformers import AutoModelForCausalLM
import torch
import os

from pathlib import Path
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
import visio_gptq.download_model as downloader
from visio_gptq import find_layers # type: ignore
from visio_gptq import make_quant # type: ignore
CURRENT_PATH = Path(__file__)

def load_quant(model, checkpoint, wbits, groupsize=-1, faster_kernel=False, exclude_layers=['lm_head'], kernel_switch_threshold=128):
    config = AutoConfig.from_pretrained(model)
    def noop(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = noop 
    torch.nn.init.uniform_ = noop 
    torch.nn.init.normal_ = noop 

    torch.set_default_dtype(torch.half)
    transformers.modeling_utils._init_weights = False
    torch.set_default_dtype(torch.half)
    model = AutoModelForCausalLM.from_config(config)
    torch.set_default_dtype(torch.float)
    model = model.eval()
    layers = find_layers(model)
    for name in exclude_layers:
        if name in layers:
            del layers[name]
    make_quant(model, layers, wbits, groupsize, faster=faster_kernel, kernel_switch_threshold=kernel_switch_threshold)

    del layers

    print('Loading model ...')
    if checkpoint.endswith('.safetensors'):
        from safetensors.torch import load_file as safe_load
        model.load_state_dict(safe_load(checkpoint))
    else:
        model.load_state_dict(torch.load(checkpoint))
    model.seqlen = 2048
    print('Done.')

    return model


def load_quantized(model_name, wbits=4, groupsize=128, threshold=128):
    if not os.path.exists(model_name):
        new_model_name = model_name.replace('/', '_')
        path_to_model = Path(f'{CURRENT_PATH.parent}/models/{new_model_name}')
        if not os.path.exists(path_to_model):
            # Cleaning up the model/branch names
            model, branch = downloader.sanitize_model_and_branch_names(model_name, 'main')
             # Getting the download links from Hugging Face
            links, sha256, is_lora = downloader.get_download_links_from_huggingface(model, branch)
            # Getting the output folder
            #path_to_model = downloader.get_output_folder(model, branch, is_lora)
            os.mkdir(path_to_model)
            downloader.download_model_files(model,branch=branch, links=links, sha256=sha256, output_folder=path_to_model)
    else:
        path_to_model = Path(model_name)
    found_pts = list(path_to_model.glob("*.pt"))
    found_safetensors = list(path_to_model.glob("*.safetensors"))
    pt_path = None

    if len(found_pts) == 1:
        pt_path = found_pts[0]
    elif len(found_safetensors) == 1:
        pt_path = found_safetensors[0]

    if not pt_path:
        print("Could not find the quantized model in .pt or .safetensors format, exiting...")
        exit()

    model = load_quant(str(path_to_model), str(pt_path), wbits, groupsize, kernel_switch_threshold=threshold)

    return model
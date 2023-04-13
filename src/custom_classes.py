from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

import dataclasses
from enum import auto, Enum
from typing import List
from pathlib import Path
import transformers
from transformers import AutoConfig, AutoModelForCausalLM
from modelutils import find_layers
from quant import make_quant


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
    model_name = model_name.replace('/', '_')
    path_to_model = Path(f'./models/{model_name}')
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

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()

@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str | None = None

    skip_next: bool = False

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system + self.sep
            for role, message in self.messages:
                if message:
                    ret += role + ": " + message + self.sep
                else:
                    ret += role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])
    
    @property
    def last_message(self):
        return self.messages[-1][1]

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2)

    def clear_messages(self):
        self.messages.clear()
    
    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
        }


class GPTQModel:
    def __init__(self, model_name, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs
        
        self.device = kwargs.get("device", "cuda")
        self.num_gpus = kwargs.get("num_gpus", 1) if self.device == "cuda" else 0
        self.wbits = kwargs.get("wbits", 0)
        self.groupsize = kwargs.get("groupsize", 128)
        
        tokenizer, model, context_len = self.__load_model(model_name=self.model_name, num_gpus=self.num_gpus, wbits=self.wbits, groupsize=self.groupsize)
        self.model = model
        self.context_len = context_len
        self.tokenizer = tokenizer


    def __load_model(self, model_name, num_gpus, wbits, groupsize):
        if num_gpus == 1:
            kwargs = {}
        else:
            kwargs = {
                "device_map": "auto",
                "max_memory": {i: "13GiB" for i in range(num_gpus)},
            }

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        if wbits > 0:
            model = load_quantized(model_name, wbits=wbits, groupsize=groupsize)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                **kwargs
            )

        if num_gpus == 1:
            model.cuda()

        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, context_len
    
    @torch.inference_mode()
    def __generate_stream(self, params, context_len=2048, stream_interval=2):

        prompt = params["prompt"]
        l_prompt = len(prompt)
        temperature = float(params.get("temperature") or 0)
        max_new_tokens = int(params.get("max_new_tokens") or 512)
        stop_str = params.get("stop", None)

        input_ids = self.tokenizer(prompt).input_ids
        output_ids = list(input_ids)

        max_src_len = context_len - max_new_tokens - 8
        input_ids = input_ids[-max_src_len:]
        past_key_values = None
        token = None
        for i in range(max_new_tokens):
            if i == 0:
                out = self.model(torch.as_tensor([input_ids], device=self.device), use_cache=True)
                logits = out.logits
                past_key_values = out.past_key_values
            else:
                if past_key_values is None:
                    raise ValueError("past_key_values is None")
                attention_mask = torch.ones(
                    1, past_key_values[0][0].shape[-2] + 1, device=self.device)
                out = self.model(input_ids=torch.as_tensor([[token]], device=self.device),
                            use_cache=True,
                            attention_mask=attention_mask,
                            past_key_values=past_key_values)
                logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]
            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token == self.tokenizer.eos_token_id:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                output = self.tokenizer.decode(output_ids, skip_special_tokens=True)
                pos = output.rfind(stop_str, l_prompt)
                if pos != -1:
                    output = output[:pos]
                    stopped = True
                yield output

            if stopped:
                break

        del past_key_values

    def inference(self, params):
        conversation = params.get("conversation", None)
        if conversation is None:
            raise ValueError("conversation is None")
        if not isinstance(conversation, Conversation):
            raise ValueError("conv_template is not a ConversationTemplate")

        inp = params.get("prompt", None)
        conversation.append_message(conversation.roles[0], inp)
        conversation.append_message(conversation.roles[1], None)
        prompt = conversation.get_prompt()

        params["prompt"] = prompt
        params["stop"] = conversation.sep if conversation.sep_style == SeparatorStyle.SINGLE else conversation.sep2

        outputs = []
        for outputs in self.__generate_stream(params):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
        conversation.messages[-1][-1] = " ".join(outputs)
        return conversation
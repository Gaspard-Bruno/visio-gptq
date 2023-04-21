from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from transformers import  AutoModelForCausalLM
from src.conversation import Conversation, SeparatorStyle

from src.quad import load_quantized


class GPTQModel:
    def __init__(self, model_name:str, **kwargs):
        self.model_name = model_name
        self.kwargs = kwargs

        self.device = kwargs.get("device", "cuda")
        self.num_gpus = kwargs.get("num_gpus", 1) if self.device == "cuda" else 0
        self.wbits = kwargs.get("wbits", 0)
        self.groupsize = kwargs.get("groupsize", 128)

        tokenizer, model, context_len = self.__load_model(model_name=self.model_name, device=self.device, num_gpus=self.num_gpus, wbits=self.wbits, groupsize=self.groupsize)
        self.model = model
        self.context_len = context_len
        self.tokenizer = tokenizer


    def __load_model(self, model_name, device, num_gpus, wbits, groupsize, max_gpu_memory="13GiB", **kwargs):

        if device == "cpu":
            kwargs = {}
        elif device == "cuda":
            kwargs = {"torch_dtype": torch.float16}
            if num_gpus == "auto":
                kwargs.update({"device_map": "auto"}) # type: ignore
            else:
                num_gpus = int(num_gpus)
                if num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: max_gpu_memory for i in range(num_gpus)},
                    })# type: ignore
        else:
            raise ValueError(f"Invalid device: {device}")

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
    def __generate_stream(self, prompt:str, **kwargs):

        l_prompt = len(prompt)
        temperature = float(kwargs.get("temperature") or 0)
        max_new_tokens = int(kwargs.get("max_new_tokens") or 512)
        context_len = int(kwargs.get("context_len") or 2048)
        stream_interval = int(kwargs.get("stream_interval") or 2)
        stop_array = kwargs.get("stop") or ["###"]

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
                for stop_str in stop_array:
                    pos = output.rfind(stop_str, l_prompt)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                        break
                yield output

            if stopped:
                break

        del past_key_values

    def inference(self, prompt: str, **kwargs):
        conversation = kwargs.get("conversation", None)
        if conversation is not None:
            if not isinstance(conversation, Conversation):
                raise ValueError("conv_template is not a ConversationTemplate")

            conversation.append_message(conversation.roles[0], prompt)
            conversation.append_message(conversation.roles[1], None)
            new_prompt = conversation.get_prompt()
            prompt = new_prompt
            kwargs["stop"] = conversation.sep if conversation.sep_style == SeparatorStyle.SINGLE else conversation.sep2

        stream_callback = kwargs.get("stream_callback", None)
        outputs = []
        pre = 0
        for outputs in self.__generate_stream(prompt, **kwargs):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")
            
            if stream_callback:
                now = len(outputs) - 1
                if now > pre:
                    stream_callback(" ".join(outputs[pre:now]))
                    pre = now
                
        if stream_callback:
            stream_callback(" ".join(outputs[pre:]))
        if conversation is not None:
            conversation.messages[-1][-1] = " ".join(outputs)
            return conversation
        return " ".join(outputs)
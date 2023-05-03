from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import gc
from transformers import  AutoModelForCausalLM
from visio_gptq.conversation import Conversation, SeparatorStyle
from visio_gptq.quad import load_quantized


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

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=wbits==0)

        if wbits > 0:
            model = load_quantized(model_name, wbits=wbits, groupsize=groupsize)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                **kwargs
            )

        if num_gpus == 1:
            model.to(torch.device("cuda:0"))

        model.eval()
        if hasattr(model.config, "max_sequence_length"):
            context_len = model.config.max_sequence_length
        else:
            context_len = 2048

        return tokenizer, model, context_len


    @torch.inference_mode()
    def __generate_stream(self, prompt:str, **kwargs):

        len_prompt = len(prompt)
        temperature = float(kwargs.get("temperature") or 0.7)
        max_new_tokens = int(kwargs.get("max_new_tokens") or 2048)
        context_len = int(kwargs.get("context_len") or 2048)
        stream_interval = int(kwargs.get("stream_interval") or 2)
        stop_array = kwargs.get("stop") or ["###"]
        echo = kwargs.get("echo") or True
        stop_token_ids = kwargs.get("stop_token_ids", []) or []
        stop_token_ids.append(self.tokenizer.eos_token_id)

        input_ids = self.tokenizer(prompt).input_ids
        input_echo_len = len(input_ids)
        output_ids = list(input_ids)

        if self.model.config.is_encoder_decoder:
            max_src_len = context_len
        else:
            max_src_len = context_len - max_new_tokens - 8

        input_ids = input_ids[-max_src_len:]
        start_ids = None
        encoder_output = None
        token = None
        past_key_values = None
        out = None
        if self.model.config.is_encoder_decoder:
            encoder_output = self.model.encoder(input_ids=torch.as_tensor([input_ids],
                                                        device=self.device))[0]
            start_ids = torch.as_tensor([[self.model.generation_config.decoder_start_token_id]],
                        dtype=torch.int64, device=self.device)

        for i in range(max_new_tokens):
            if i == 0:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(input_ids=start_ids,
                                        encoder_hidden_states=encoder_output,
                                        use_cache=True)
                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(torch.as_tensor([input_ids], device=self.device), use_cache=True)
                    logits = out.logits
                past_key_values = out.past_key_values
            else:
                if self.model.config.is_encoder_decoder:
                    out = self.model.decoder(input_ids=torch.as_tensor([[token]],
                                            device=self.device),
                                             encoder_hidden_states=encoder_output,
                                             use_cache=True,
                                             past_key_values=past_key_values)

                    logits = self.model.lm_head(out[0])
                else:
                    out = self.model(
                        input_ids=torch.as_tensor([[token]],
                        device=self.device),
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    logits = out.logits
                past_key_values = out.past_key_values

            last_token_logits = logits[0][-1]

            if self.device == "mps":
                # Switch to CPU by avoiding some bugs in mps backend.
                last_token_logits = last_token_logits.float().to("cpu")

            if temperature < 1e-4:
                token = int(torch.argmax(last_token_logits))
            else:
                probs = torch.softmax(last_token_logits / temperature, dim=-1)
                token = int(torch.multinomial(probs, num_samples=1))

            output_ids.append(token)

            if token in stop_token_ids:
                stopped = True
            else:
                stopped = False

            if i % stream_interval == 0 or i == max_new_tokens - 1 or stopped:
                if echo:
                    tmp_output_ids = output_ids
                    rfind_start = len_prompt
                else:
                    tmp_output_ids = output_ids[input_echo_len:]
                    rfind_start = 0

                output = self.tokenizer.decode(tmp_output_ids, skip_special_tokens=True, 
                                        spaces_between_special_tokens=False)
                for stop_str in stop_array:
                    pos = output.rfind(stop_str, rfind_start)
                    if pos != -1:
                        output = output[:pos]
                        stopped = True
                yield output

            if stopped:
                break

        del past_key_values, out
        gc.collect()
        torch.cuda.empty_cache()

    def generate_stream(self, prompt: str, **kwargs):
        conversation = kwargs.get("conversation", None)
        if conversation is not None:
            if not isinstance(conversation, Conversation):
                raise ValueError("conv_template is not a ConversationTemplate")

            conversation.append_message(conversation.roles[0], prompt)
            conversation.append_message(conversation.roles[1], None)
            new_prompt = conversation.get_prompt()
            prompt = new_prompt
            kwargs["stop"] = conversation.sep if conversation.sep_style == SeparatorStyle.SINGLE else conversation.sep2

        outputs = []
        pre = 0
        for outputs in self.__generate_stream(prompt, **kwargs):
            outputs = outputs[len(prompt) + 1:].strip()
            outputs = outputs.split(" ")

            now = len(outputs) - 1
            if now > pre:
                data = " ".join(outputs[pre:now])
                yield data
                pre = now
        yield " ".join(outputs[pre:])

    def generate(self, prompt: str, **kwargs):
        generator = self.generate_stream(prompt=prompt, **kwargs)
        stream_callback = kwargs.get("stream_callback", None)
        conversation = kwargs.get("conversation", None)
        output = ""
        for result in generator:
            output += " " + result
            if stream_callback:
                stream_callback(result)

        if conversation is not None:
            conversation.messages[-1][-1] = output
            return conversation
        return output
    
    def inference(self, prompt: str, **kwargs):
        return self.generate(prompt, **kwargs)
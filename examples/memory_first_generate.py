import torch
from llm_memlab import MemoryFirstHFConfig, memory_first_generate


class TinyHF(torch.nn.Module):
    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 8)
        logits[..., 1] = 1.0
        k = torch.zeros(input_ids.shape[0], 1, input_ids.shape[1], 4)
        v = torch.zeros_like(k)
        return {"logits": logits, "past_key_values": ((k, v),)}


if __name__ == "__main__":
    result = memory_first_generate(TinyHF(), torch.tensor([[1, 2]]), MemoryFirstHFConfig(max_new_tokens=3, cache="quantized"))
    print(result.to_text())

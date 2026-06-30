import torch
from llm_memlab import assert_quality_regression, run_quality_regression


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 8)
        self.proj = torch.nn.Linear(8, 16)

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        logits = self.proj(self.embed(input_ids))
        k = torch.zeros(input_ids.shape[0], 1, input_ids.shape[1], 8)
        return {"logits": logits, "past_key_values": ((k, k),)}


if __name__ == "__main__":
    baseline = TinyLM().eval()
    candidate = TinyLM().eval()
    candidate.load_state_dict(baseline.state_dict())
    result = run_quality_regression(baseline, candidate, torch.tensor([[1, 2, 3]]))
    print(result.to_text())
    assert_quality_regression(result)

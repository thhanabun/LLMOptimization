import torch
from llm_memlab import compare_layer_drift

baseline = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)).eval()
candidate = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)).eval()
candidate.load_state_dict(baseline.state_dict())
with torch.no_grad():
    candidate[2].bias.add_(0.01)

report = compare_layer_drift(baseline, candidate, torch.randn(2, 4, 8))
print(report.to_text())

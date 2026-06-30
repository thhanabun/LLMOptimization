import torch
from llm_memlab import build_deep_debug_report, write_deep_debug_html


if __name__ == "__main__":
    baseline = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)).eval()
    candidate = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 8)).eval()
    candidate.load_state_dict(baseline.state_dict())
    with torch.no_grad():
        candidate[2].bias.add_(0.001)
    report = build_deep_debug_report(baseline, candidate, torch.randn(2, 8))
    path = write_deep_debug_html(report, "deep_debug_example.html")
    print(f"wrote {path}")

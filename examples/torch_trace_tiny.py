import torch

from llm_memlab.torch_debugger import TorchTrace


model = torch.nn.Sequential(
    torch.nn.Linear(16, 64),
    torch.nn.GELU(),
    torch.nn.Linear(64, 16),
)
x = torch.randn(8, 16)

with TorchTrace(model) as trace:
    y = model(x)

print(trace.to_text())

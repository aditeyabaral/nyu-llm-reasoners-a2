import torch
import torch.nn as nn


class ToyModel(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        print(f"fc1 weight dtype:     {self.fc1.weight.dtype}")
        x = self.relu(self.fc1(x))
        print(f"fc1 output dtype:     {x.dtype}")
        x = self.ln(x)
        print(f"layer norm output:    {x.dtype}")
        x = self.fc2(x)
        print(f"logits dtype:         {x.dtype}")
        return x


model = ToyModel(20, 5).cuda()
x = torch.randn(4, 20).cuda()

with torch.autocast(device_type="cuda", dtype=torch.float16):
    logits = model(x)
    loss = logits.sum()
    print(f"loss dtype:           {loss.dtype}")

loss.backward()
print(f"fc1 weight grad dtype: {model.fc1.weight.grad.dtype}")

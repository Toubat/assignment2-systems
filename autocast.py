from modal import App, Image

app = App("autocast")

image = Image.debian_slim().pip_install("torch")

import torch.nn as nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        print("fc1", x.dtype)
        x = self.ln(x)
        print("ln", x.dtype)
        x = self.fc2(x)
        print("fc2", x.dtype)
        return x

@app.function(image=image, gpu="L4", timeout=60 * 30)
def autocast():
    import torch

    model = ToyModel(10, 10).to("cuda")
    x = torch.randn(10, 10, dtype=torch.float32).to("cuda")
    target = torch.randint(0, 10, (10,), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, target)
        print("loss", loss.dtype)

    # backward goes OUTSIDE the autocast context (standard AMP pattern).
    loss.backward()

    print("logits", logits.dtype)
    # Gradients live on leaf tensors (parameters), not on the non-leaf `logits`.
    print("fc1.weight.grad", model.fc1.weight.grad.dtype)

@app.local_entrypoint()
def main():
    print("returned:", autocast.remote())
"""
export_onnx.py — Export the deployed ensemble to ONNX for the in-browser demo.

    python deploy/export_onnx.py

Writes deploy/web/insightvlm_ensemble.onnx: input "image" (1x3x384x384, ImageNet-
normalised), output "probs" (members x 20 softmax). The page averages the members
with equal weights, exactly as results/final_ensemble.json specifies (alpha = 0).
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "train"))
from build_ensemble import build  # noqa: E402

OUT = ROOT / "deploy" / "web"


class Members(nn.Module):
    def __init__(self, nets):
        super().__init__()
        self.nets = nn.ModuleList(nets)

    def forward(self, x):
        return torch.stack([torch.softmax(n(x), 1)[0] for n in self.nets])


def main():
    cfg = json.loads((ROOT / "results" / "final_ensemble.json").read_text())
    assert cfg["alpha"] == 0, "the page assumes no label-shift correction"
    sizes = {m["size"] for m in cfg["members"]}
    assert sizes == {384}, sizes
    nets = []
    for m in cfg["members"]:
        n = build(m["arch"], len(cfg["classes"]))
        n.load_state_dict(torch.load(ROOT / m["ckpt"], map_location="cpu"))
        nets.append(n.eval())
    model = Members(nets).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.randn(1, 3, 384, 384), OUT / "insightvlm_ensemble.onnx",
                      input_names=["image"], output_names=["probs"], opset_version=17, dynamo=False)
    (OUT / "model.json").write_text(json.dumps({
        "classes": cfg["classes"], "members": [m["name"] for m in cfg["members"]],
        "size": 384, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225],
        "test": cfg["test"]}, indent=2))
    print("wrote", OUT / "insightvlm_ensemble.onnx")


if __name__ == "__main__":
    main()

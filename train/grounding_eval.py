"""
grounding_eval.py — RQ3 / O4: does the model look at the evidence?

    python train/grounding_eval.py

Pointing game on the real test charts. For each chart with at least one evidence
box (modal bin, outlier band, change point, anomaly points, dominant wedge, ...),
take the Grad-CAM peak of the deployed members for the predicted class, map it back
to image pixels, and count a hit if it falls inside any evidence box. The control
is a uniformly random point inside the plot area, whose expected hit rate is the
evidence boxes' share of the plot area, so large boxes cannot inflate the score.

Writes results/grounding.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_ensemble import DATA, DEV, RES, build, tf_for  # noqa: E402

W, H = 768, 576


def features_and_logits(net, arch, x):
    """Explicit forward pass returning the last feature map (grad retained) and logits.
    torchvision's DenseNet applies an in-place ReLU right after its last block, which
    module backward hooks forbid, so no hooks are used."""
    if arch.startswith("resnet"):
        h = net.maxpool(net.relu(net.bn1(net.conv1(x))))
        a = net.layer4(net.layer3(net.layer2(net.layer1(h))))
        logits = net.fc(torch.flatten(net.avgpool(a), 1))
    else:
        a = F.relu(net.features(x))
        logits = net.classifier(torch.flatten(F.adaptive_avg_pool2d(a, 1), 1))
    a.retain_grad()
    return a, logits


def gradcam(net, arch, x):
    net.zero_grad()
    a, out = features_and_logits(net, arch, x)
    c = int(out.argmax(1))
    out[0, c].backward()
    g = a.grad[0]
    cam = F.relu((g.mean((1, 2), keepdim=True) * a[0]).sum(0))
    return cam.detach().cpu().numpy(), c


def union_area(boxes, plot):
    """Fraction of the plot area covered by the evidence boxes (their union, rasterised)."""
    x0, y0, x1, y1 = [int(round(v)) for v in plot]
    m = np.zeros((H, W), bool)
    for b in boxes:
        a, bb, c, d = [int(round(v)) for v in b]
        m[max(bb, y0):min(d, y1), max(a, x0):min(c, x1)] = True
    area = max((x1 - x0) * (y1 - y0), 1)
    return m[y0:y1, x0:x1].sum() / area


def main():
    cfg = json.loads((RES / "final_ensemble.json").read_text())
    classes = cfg["classes"]
    meta = pd.read_parquet(DATA / "tables" / "metadata.parquet")
    te = meta[(meta.split == "test") & (meta.provenance == "real")].reset_index(drop=True)
    rows = []
    for mem in cfg["members"]:
        net = build(mem["arch"], len(classes))
        net.load_state_dict(torch.load(DATA.parent / mem["ckpt"], map_location="cpu"))
        net = net.to(DEV).eval()
        tf, S = tf_for(mem["size"]), mem["size"]
        for r in te.itertuples():
            boxes = json.loads(r.grounding_boxes_json)
            plot = next((b["bbox"] for b in boxes if b["label"] == "plot_area"), None)
            ev = [b["bbox"] for b in boxes if b["label"] != "plot_area"]
            if plot is None or not ev:
                continue
            img = Image.open(DATA / r.image).convert("RGB")
            cam, c = gradcam(net, mem["arch"], tf(img).unsqueeze(0).to(DEV))
            cam = F.interpolate(torch.tensor(cam)[None, None], size=(S, S), mode="bilinear",
                                align_corners=False)[0, 0].numpy()
            py, px = np.unravel_index(cam.argmax(), cam.shape)
            x, y = (px + 0.5) * W / S, (py + 0.5) * H / S     # back to image pixels
            hit = any(b[0] <= x <= b[2] and b[1] <= y <= b[3] for b in ev)
            rows.append({"member": mem["name"], "sample_id": r.sample_id, "label": r.label,
                         "family": r.family, "correct": classes[c] == r.label, "hit": hit,
                         "chance": union_area(ev, plot)})
        print(f"{mem['name']}: {sum(1 for x in rows if x['member'] == mem['name'])} charts", flush=True)
    d = pd.DataFrame(rows)
    out = {"n_charts": int(d.sample_id.nunique()),
           "by_member": {m: {"pointing": float(g.hit.mean()), "random_control": float(g.chance.mean()),
                             "pointing_correct_only": float(g[g.correct].hit.mean())}
                         for m, g in d.groupby("member")},
           "by_family": {f: {"pointing": float(g.hit.mean()), "random_control": float(g.chance.mean()),
                             "n": int(g.sample_id.nunique())} for f, g in d.groupby("family")}}
    out["overall"] = {"pointing": float(d.hit.mean()), "random_control": float(d.chance.mean())}
    (RES / "grounding.json").write_text(json.dumps(out, indent=2))
    d.to_csv(RES / "grounding_rows.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

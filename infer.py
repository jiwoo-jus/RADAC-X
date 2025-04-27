import sys, pathlib, torch, pandas as pd, numpy as np
from PIL import Image
from open_clip import create_model_from_pretrained
import torch.nn as nn

USAGE = "python infer.py <variant {v1|v2|v3|v4}> <csv_path>"

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation",
          "Edema", "Pleural Effusion"]

# absolute paths on the cluster
ABS = {
    "v1": "/fs/scratch/PAS2985/group2/models/linear_classifier_original.pt",
    "v2": "/fs/scratch/PAS2985/group2/models/linear_classifier_masked.pt",
    "v3": "/fs/scratch/PAS2985/group2-w/folder_to_share/best_model.pt",
    "v4": "/fs/scratch/PAS2985/group2/models/vision_encoder_best.pt",
}
# fallback local filenames
LOCAL = {
    "v1": "checkpoints/v1_classifier_original.pt",
    "v2": "checkpoints/v2_classifier_overlay.pt",
    "v3": "checkpoints/v3_classifier_and_vision_encoder_original.pt",
    "v4": "checkpoints/v4_classifier_and_vision_encoder_overlay.pt",
}

# ---------------------------------------------------------------- helpers
def load_ckpt(variant, dev):
    """Try absolute path first, then local file; return loaded checkpoint."""
    for p in (ABS[variant], LOCAL[variant]):
        path = pathlib.Path(p)
        if path.exists():
            return torch.load(path, map_location=dev)
    raise FileNotFoundError(f"Checkpoint for {variant} not found.")

def extract_head(sd):
    """Keep only the classifier head weight & bias and rename keys -> weight/bias."""
    head_sd = {}
    for w_key in ("head.weight", "classifier.weight", "weight"):
        if w_key in sd:
            head_sd["weight"] = sd[w_key]
            break
    for b_key in ("head.bias", "classifier.bias", "bias"):
        if b_key in sd:
            head_sd["bias"] = sd[b_key]
            break
    if len(head_sd) != 2:
        raise ValueError("Head weights not found in checkpoint.")
    return head_sd

def rename_classifier(sd):
    """Convert 'classifier.*' keys to 'head.*' for full-model checkpoints."""
    out = {}
    for k, v in sd.items():
        if k.startswith("classifier."):
            out["head" + k[len("classifier"):]] = v
        else:
            out[k] = v
    return out

# ---------------------------------------------------------------- wrapper
class CLIPClassifier(nn.Module):
    """BiomedCLIP vision encoder + custom linear head."""
    def __init__(self, base, n=5):
        super().__init__()
        self.visual = base.visual
        with torch.no_grad():
            dim = self.visual(torch.randn(1, 3, 224, 224)).shape[-1]
        self.head = nn.Linear(dim, n)

    def forward(self, x):
        return self.head(self.visual(x))

# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ABS:
        print(USAGE)
        sys.exit(1)

    variant, csv_path = sys.argv[1], sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) load BiomedCLIP backbone & tokenizer-compatible preprocess
    base_clip, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )
    model = CLIPClassifier(base_clip).to(device)

    # 2) load checkpoint
    ckpt = load_ckpt(variant, device)
    sd = ckpt["model_state"] if "model_state" in ckpt else ckpt

    if variant in ("v1", "v2"):
        model.head.load_state_dict(extract_head(sd), strict=True)
    else:
        model.load_state_dict(rename_classifier(sd), strict=False)

    model.eval()

    # 3) run inference on images listed in CSV
    df = pd.read_csv(csv_path)
    for i, row in df.iterrows():
        img = preprocess(Image.open(row["imgpath"]).convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = model(img).softmax(-1).squeeze().cpu().numpy()
        pred = LABELS[int(probs.argmax())]

        print(f"[{i}] {row['imgpath']}")
        print(f"    GT : {row['class']}")
        print(f"    PR : {pred}  (prob={probs.max():.2f})")
        print("    probs:", {l: round(float(v), 3) for l, v in zip(LABELS, probs)})

if __name__ == "__main__":
    main()

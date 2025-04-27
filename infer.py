import sys, pathlib, torch, pandas as pd
from PIL import Image
from open_clip import create_model_from_pretrained
import torch.nn as nn

USAGE = "python infer.py <variant {v1|v2|v3|v4}> <csv_path>"

LABELS = ["Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion"]

# absolute paths on the cluster
ABS = {
    "v1": "/fs/scratch/PAS2985/group2/fine-tuned-models/classifier_original.pt",
    "v2": "/fs/scratch/PAS2985/group2/fine-tuned-models/classifier_masked.pt",
    "v3": "/fs/scratch/PAS2985/group2-w/folder_to_share/best_model.pt",
    "v4": "/fs/scratch/PAS2985/group2/fine-tuned-models/vision_encoder_masked.pt",
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

# ---------------------------------------------------------------- wrapper
class BioMedCLIPClassifier(nn.Module):
    """BiomedCLIP vision encoder + custom linear head."""
    def __init__(self, base_model, num_classes, device):
        super().__init__()
        self.visual = base_model.visual.to(device)
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224).to(device)
            feat = self.visual(dummy)
            feat_dim = feat.shape[-1]
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        feat = self.visual(x)
        return self.classifier(feat)

# ---------------------------------------------------------------- main
def main():
    if len(sys.argv) != 3 or sys.argv[1] not in ABS:
        print(USAGE)
        sys.exit(1)

    variant, csv_path = sys.argv[1], sys.argv[2]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Load BiomedCLIP backbone & preprocess
    base_clip, preprocess = create_model_from_pretrained(
        "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    )

    # 2) Initialize model
    model = BioMedCLIPClassifier(base_clip, num_classes=5, device=device)

    # 3) Load checkpoint
    ckpt = load_ckpt(variant, device)
    if variant in ("v1", "v2"):
        model.classifier.load_state_dict(ckpt["model_state"], strict=True)
    else:
        model.load_state_dict(ckpt["model_state"], strict=True)

    model.to(device).eval()

    # 4) Run inference on images listed in CSV
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

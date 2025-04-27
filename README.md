# RADAC-X Demo

```
radac_x_demo/
├─ checkpoints/       # *.pt files (download if the cluster paths are inaccessible)
├─ samples/           # 10 demo images (5 classes × original / overlay)
├─ data.csv           # list of images the script will score
├─ infer.py           # single-file inference driver
└─ environment.yml    # conda recipe (CUDA 11.8 · PyTorch 2.6.0)
```

---

## 1 · Setup

```bash
# create and activate the environment
conda env create -f environment.yml
conda activate radac_x_demo
```

> The supplied `environment.yml` is tailored to an NVIDIA A100 GPU with CUDA 11.8.  
> If you run on a different accelerator, use the matching
> PyTorch wheels from <https://pytorch.org/get-started/previous-versions/>.

---

## 2 · Model variants

| ID | Fine-tuned parameters | Training images |
|----|----------------------|-----------------|
| **v1** | classifier head only | original chest X-rays |
| **v2** | classifier head only | overlay images (lung & heart contours) |
| **v3** | classifier + vision encoder | original chest X-rays |
| **v4** | classifier + vision encoder | overlay images |

> **Checkpoints**  
> *If you are registered on the OSC PAS2985 cluster*, the absolute paths in `infer.py` (`ABS` dict) will load automatically.  
> *If you are off the cluster*, download or copy the four checkpoints into the `checkpoints/` directory, ensuring the filenames match those referenced in the script.

---

## 3 · Running inference

```bash
python infer.py {v1|v2|v3|v4} data.csv
```

The script prints, for each image:

```
[0] samples/Cardiomegaly_original.jpg
    Ground truth     : Cardiomegaly
    Model prediction : Cardiomegaly  (prob=0.95)
    probs            : {'Atelectasis': 0.03, 'Cardiomegaly': 0.95, ...}
```

---

## 4 · data.csv

Two required columns:

```csv
class,imgpath
Cardiomegaly,samples/Cardiomegaly_original.jpg
Cardiomegaly,samples/Cardiomegaly_overlay.jpg
Atelectasis,samples/Atelectasis_original.jpg
...
```

* **class** – one of `Atelectasis | Cardiomegaly | Consolidation | Edema | Pleural Effusion`
* **imgpath** – relative or absolute path to image.

You can swap in any image list—just keep the header names identical.

---

## 5 · Sample images

`/samples` already contains 10 resized 224 × 224 JPEGs:

* `*_original.jpg` – raw frontal views from **MIMIC-CXR-JPG**  
* `*_overlay.jpg` – the same image with a thin contour (“overlay mask”) drawn around
  left lung, right lung, and heart using **CheXmask** segmentation.

They correspond to the rows in the default `data.csv`, so all four variants run
out-of-the-box.

---

## 6 · Training data sources

* **MIMIC-CXR-JPG** – large-scale chest radiograph dataset  
  <https://physionet.org/content/mimic-cxr-jpg/>
* **CheXmask** – anatomical segmentation masks for MIMIC-CXR images  
  <https://physionet.org/content/chexmask-cxr-segmentation-data/1.0.0/>

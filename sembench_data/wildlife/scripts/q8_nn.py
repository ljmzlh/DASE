"""
Wildlife Q8 — DASE-only (no BigQuery): cities with elephant (image OR audio) AND
monkey (image OR audio).

NL: (img_eleph ∪ aud_eleph) ∩ (img_monk ∪ aud_monk) over City.
Eval: set retrieval F1 (matches original).

Aligns with paper §5.1: counterfactual anchors per (modality × concept). Anchors
verbatim from q8_cascade.py (4 prompt sets). Four MarginSignal passes; client-side
union-then-intersect.
"""
import os
import sys

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from dase_cascade import MarginSignal

WILDLIFE_DIR = os.path.abspath(os.path.join(_HERE, ".."))
IMAGE_CSV    = os.path.join(WILDLIFE_DIR, "cache", "image_data.csv")
AUDIO_CSV    = os.path.join(WILDLIFE_DIR, "cache", "audio_data.csv")
IMG_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "image_embeddings.npz")
AUD_EMB_PATH = os.path.join(WILDLIFE_DIR, "data", "audio_embeddings.npz")

IMG_ELEPH_POS = ["a photograph of an elephant", "a wildlife camera trap image showing an elephant", "an elephant captured in the photo"]
IMG_ELEPH_NEG = ["a photograph that does not contain an elephant", "a wildlife camera trap image of a non-elephant animal", "an animal photo without any elephant"]
AUD_ELEPH_POS = ["a sound recording of an elephant", "audio of an elephant trumpeting or vocalizing", "elephant call sound clip"]
AUD_ELEPH_NEG = ["a sound recording of an animal that is not an elephant", "audio of a non-elephant animal vocalization", "animal sound clip without any elephant"]
IMG_MONK_POS = ["a photograph of a monkey", "a wildlife camera trap image showing a monkey", "a monkey captured in the photo"]
IMG_MONK_NEG = ["a photograph that does not contain a monkey", "a wildlife camera trap image of a non-monkey animal", "an animal photo without any monkey"]
AUD_MONK_POS = ["a sound recording of a monkey", "audio of monkey vocalizations or calls", "monkey howling or chittering sound clip"]
AUD_MONK_NEG = ["a sound recording of an animal that is not a monkey", "audio of a non-monkey animal vocalization", "animal sound clip without any monkey"]


def main():
    df_i = pd.read_csv(IMAGE_CSV)
    df_a = pd.read_csv(AUDIO_CSV)
    img_emb = np.load(IMG_EMB_PATH)["caption_emb"]
    aud_emb = np.load(AUD_EMB_PATH)["caption_emb"]

    img_eleph_m = MarginSignal(positive_prompts=IMG_ELEPH_POS, negative_prompts=IMG_ELEPH_NEG).compute(img_emb)
    aud_eleph_m = MarginSignal(positive_prompts=AUD_ELEPH_POS, negative_prompts=AUD_ELEPH_NEG).compute(aud_emb)
    img_monk_m  = MarginSignal(positive_prompts=IMG_MONK_POS,  negative_prompts=IMG_MONK_NEG).compute(img_emb)
    aud_monk_m  = MarginSignal(positive_prompts=AUD_MONK_POS,  negative_prompts=AUD_MONK_NEG).compute(aud_emb)

    el_cities = set(df_i.loc[img_eleph_m > 0, "City"]) | set(df_a.loc[aud_eleph_m > 0, "City"])
    mk_cities = set(df_i.loc[img_monk_m  > 0, "City"]) | set(df_a.loc[aud_monk_m  > 0, "City"])
    pred = sorted(el_cities & mk_cities)

    gt_el = (
        set(df_i.loc[df_i["Species"].str.contains("ELEPHANT", case=False, na=False), "City"])
        | set(df_a.loc[df_a["Animal"] == "Elephant", "City"])
    )
    gt_mk = (
        set(df_i.loc[df_i["Species"].str.contains("MONKEY", case=False, na=False), "City"])
        | set(df_a.loc[df_a["Animal"] == "Monkey", "City"])
    )
    gt = sorted(gt_el & gt_mk)

    pred_set, gt_set = set(pred), set(gt)
    tp = len(pred_set & gt_set)
    prec = tp / len(pred_set) if pred_set else (1.0 if not gt_set else 0.0)
    rec = tp / len(gt_set) if gt_set else (1.0 if not pred_set else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    print(f"Predicted: {pred}")
    print(f"Ground truth: {gt}")
    print(f"[SemBench] F1={f1:.4f}")


if __name__ == "__main__":
    main()

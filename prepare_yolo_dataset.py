import pandas as pd
import shutil
import os
from pathlib import Path
from ultralytics import YOLO
import cv2
from collections import Counter

# -------------------- CONFIGURATION --------------------
CSV_PATH = "0623_specimens_modeling_master.csv"
IMAGE_SOURCE_DIR = "./0623"          # where the original images are stored
OUTPUT_DIR = "./yolo_dataset"        # YOLO dataset will be created here
MODEL_PATH = "best.pt"               # your trained YOLO model
CONF_THRESHOLD = 0.5
TASK = "detection"                   # keep detection

# Mapping from CSV species labels to YOLO class IDs (0,1,2)
SPECIES_TO_ID = {
    "anopheles_gambiae": 0,
    "anopheles_stephensi": 1,
    "aedes_aegypti": 2
}
ID_TO_SPECIES = {v: k for k, v in SPECIES_TO_ID.items()}

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
IMG_DIR = Path(OUTPUT_DIR) / "images"
LABEL_DIR = Path(OUTPUT_DIR) / "labels"
IMG_DIR.mkdir(exist_ok=True)
LABEL_DIR.mkdir(exist_ok=True)
# -------------------------------------------------------

def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["SpeciesLabel"].isin(SPECIES_TO_ID.keys())]
    return df

def copy_images(df, source_dir, dest_dir):
    """Copy all images to images/ folder."""
    for _, row in df.iterrows():
        f = row["DownloadedFilename"]
        src = Path(source_dir) / f
        dst = Path(dest_dir) / "images" / f
        if src.exists():
            shutil.copy(src, dst)

def auto_annotate(model_path, img_dir, label_dir, conf_thres):
    """Run YOLO detection and save labels. Also collect predictions for evaluation."""
    model = YOLO(model_path)
    image_files = list(Path(img_dir).glob("*.*"))
    predictions = {}  # filename -> list of (cls_id, confidence)
    
    for img_path in image_files:
        results = model(img_path, conf=conf_thres)
        preds = []
        if results[0].boxes is not None:
            boxes = results[0].boxes.xywhn.cpu().numpy()
            cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            confs = results[0].boxes.conf.cpu().numpy()
            # Filter to keep only classes that match our mapping (optional)
            for cls_id, conf in zip(cls_ids, confs):
                if cls_id in ID_TO_SPECIES:
                    preds.append((cls_id, conf))
            # Write label file
            label_file = Path(label_dir) / (img_path.stem + ".txt")
            with open(label_file, "w") as f:
                for box, cls_id in zip(boxes, cls_ids):
                    if cls_id in ID_TO_SPECIES:
                        f.write(f"{cls_id} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")
        else:
            # empty label file
            label_file = Path(label_dir) / (img_path.stem + ".txt")
            label_file.touch()
        predictions[img_path.name] = preds
    return predictions

def evaluate_predictions(df, predictions, label_dir):
    """
    Compare predictions with ground truth from CSV.
    Returns a DataFrame with evaluation results.
    """
    # Build a map from filename -> true species
    true_map = {}
    for _, row in df.iterrows():
        fname = row["DownloadedFilename"]
        true_map[fname] = row["SpeciesLabel"]
    
    results = []
    for fname, pred_list in predictions.items():
        true_species = true_map.get(fname, "unknown")
        if not pred_list:
            pred_species = "no_detection"
            correct = False
            max_conf = "-"
        else:
            # Get the most frequent predicted class
            cls_ids = [cls_id for cls_id, _ in pred_list]
            most_common_cls = Counter(cls_ids).most_common(1)[0][0]
            pred_species = ID_TO_SPECIES.get(most_common_cls, "unknown")
            # Also get max confidence among predictions of that class
            confs = [conf for cls_id, conf in pred_list if cls_id == most_common_cls]
            max_conf = max(confs) if confs else "-"
            correct = (pred_species == true_species)
        
        results.append({
            "Filename": fname,
            "TrueSpecies": true_species,
            "PredictedSpecies": pred_species,
            "Correct": correct,
            "MaxConfidence": max_conf
        })
    
    eval_df = pd.DataFrame(results)
    return eval_df

def generate_evaluation_report(eval_df, output_csv="evaluation_report.csv"):
    eval_df.to_csv(output_csv, index=False)
    accuracy = eval_df["Correct"].mean() * 100
    print(f"Overall accuracy: {accuracy:.2f}%")
    print(f"Report saved to {output_csv}")

def create_data_yaml(output_dir, class_names):
    yaml_content = f"""
# Dataset YAML for YOLO detection
path: {output_dir}
train: images
val: images
nc: {len(class_names)}
names: {class_names}
"""
    with open(Path(output_dir) / "dataset.yaml", "w") as f:
        f.write(yaml_content)

def main():
    df = read_csv(CSV_PATH)
    print(f"Total images in CSV: {len(df)}")

    # 1. Copy images
    copy_images(df, IMAGE_SOURCE_DIR, OUTPUT_DIR)
    print("Images copied.")

    # 2. Auto‑annotate and get predictions
    print("Running YOLO detection for auto-annotation...")
    predictions = auto_annotate(MODEL_PATH, IMG_DIR, LABEL_DIR, CONF_THRESHOLD)
    print("Annotation complete.")

    # 3. Evaluate against ground truth
    eval_df = evaluate_predictions(df, predictions, LABEL_DIR)
    generate_evaluation_report(eval_df, "evaluation_report.csv")

    # 4. Create dataset.yaml
    class_names = list(SPECIES_TO_ID.keys())
    create_data_yaml(OUTPUT_DIR, class_names)
    print(f"Dataset prepared in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
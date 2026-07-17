import pandas as pd
import shutil
import os
from pathlib import Path
from ultralytics import YOLO
from collections import Counter
import cv2

# -------------------- CONFIGURATION --------------------
CSV_PATH = "0623_specimens_modeling_master.csv"
IMAGE_SOURCE_DIR = "./0623"
OUTPUT_DIR = "./yolo_dataset_corrected"
MODEL_PATH = "best.pt"
CONF_THRESHOLD = 0.3
SAVE_SPECIAL_IMAGES = True     # Save special images to a separate folder
SPECIAL_IMAGES_DIR = "./special_images"  # Where to save the annotated special images
# -------------------------------------------------------

SPECIES_TO_ID = {
    "anopheles_gambiae": 0,
    "anopheles_stephensi": 1,
    "aedes_aegypti": 2
}
ID_TO_SPECIES = {v: k for k, v in SPECIES_TO_ID.items()}

# Class colours for drawing (BGR)
CLASS_COLORS = {
    0: (0, 255, 0),    # green
    1: (0, 0, 255),    # red
    2: (255, 255, 0),  # cyan
}

# Create directories
IMG_DIR = Path(OUTPUT_DIR) / "images"
LABEL_DIR = Path(OUTPUT_DIR) / "labels"
IMG_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)
if SAVE_SPECIAL_IMAGES:
    SPECIAL_IMAGES_DIR = Path(SPECIAL_IMAGES_DIR)
    SPECIAL_IMAGES_DIR.mkdir(parents=True, exist_ok=True)


def read_csv(csv_path):
    df = pd.read_csv(csv_path)
    df = df[df["SpeciesLabel"].isin(SPECIES_TO_ID.keys())]
    return df


def copy_images(df, source_dir, dest_dir):
    for _, row in df.iterrows():
        f = row["DownloadedFilename"]
        src = Path(source_dir) / f
        dst = Path(dest_dir) / "images" / f
        if src.exists():
            shutil.copy(src, dst)
        else:
            print(f"Warning: {src} not found, skipping.")


def enforce_labels_with_csv(model_path, img_dir, label_dir, df, conf_thres):
    """
    Run YOLO detection to get boxes, assign correct species ID from CSV,
    and also return original predictions for evaluation.
    """
    model = YOLO(model_path)
    file_to_class = {}
    for _, row in df.iterrows():
        fname = row["DownloadedFilename"]
        species = row["SpeciesLabel"]
        if species in SPECIES_TO_ID:
            file_to_class[fname] = SPECIES_TO_ID[species]

    image_files = list(Path(img_dir).glob("*.*"))
    stats = {
        "total_images": len(image_files),
        "images_with_detections": 0,
        "images_no_detection": 0,
        "total_boxes": 0,
        "boxes_per_species": Counter(),
        "images_per_species": Counter(),
        "original_predictions": []  # list of (true_cls, pred_cls)
    }

    for img_path in image_files:
        fname = img_path.name
        if fname not in file_to_class:
            continue
        correct_cls = file_to_class[fname]
        stats["images_per_species"][correct_cls] += 1

        results = model(img_path, conf=conf_thres)
        label_path = Path(label_dir) / (img_path.stem + ".txt")
        boxes = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xywhn.cpu().numpy()
            pred_cls_ids = results[0].boxes.cls.cpu().numpy().astype(int)
            for pred_cls in pred_cls_ids:
                if pred_cls in ID_TO_SPECIES:
                    stats["original_predictions"].append((correct_cls, pred_cls))
            stats["images_with_detections"] += 1
            stats["total_boxes"] += len(boxes)
            stats["boxes_per_species"][correct_cls] += len(boxes)
        else:
            stats["images_no_detection"] += 1

        with open(label_path, "w") as f:
            for box in boxes:
                f.write(f"{correct_cls} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f}\n")

    return stats


def compute_confusion_matrix(predictions):
    """Compute confusion matrix from list of (true, pred) pairs."""
    if not predictions:
        return None, None
    true_vals = [t for t, p in predictions]
    pred_vals = [p for t, p in predictions]
    labels = sorted(set(true_vals + pred_vals))
    cm = pd.crosstab(pd.Series(true_vals, name="True"),
                     pd.Series(pred_vals, name="Predicted"),
                     rownames=["True"], colnames=["Predicted"])
    cm = cm.reindex(index=labels, columns=labels, fill_value=0)
    cm.index = [ID_TO_SPECIES.get(i, f"class{i}") for i in cm.index]
    cm.columns = [ID_TO_SPECIES.get(i, f"class{i}") for i in cm.columns]
    return cm


def summarize_dataset(stats, output_dir):
    print("\n========== Dataset Summary ==========")
    print(f"Total images processed:        {stats['total_images']}")
    print(f"Images with detections:        {stats['images_with_detections']}")
    print(f"Images with no detection:      {stats['images_no_detection']}")
    print(f"Total bounding boxes created:  {stats['total_boxes']}")
    print("\nPer‑species breakdown:")
    for cls_id, count in stats["images_per_species"].items():
        species = ID_TO_SPECIES.get(cls_id, f"class{cls_id}")
        box_count = stats["boxes_per_species"].get(cls_id, 0)
        print(f"  {species}: {count} images, {box_count} boxes")

    if stats["original_predictions"]:
        cm = compute_confusion_matrix(stats["original_predictions"])
        print("\nConfusion matrix (original model predictions vs ground truth):")
        print(cm)
        cm_path = Path(output_dir) / "confusion_matrix_original.csv"
        cm.to_csv(cm_path)
        print(f"Confusion matrix saved to {cm_path}")

    summary_df = pd.DataFrame({
        "Metric": ["Total images", "With detections", "No detection", "Total boxes"],
        "Value": [stats["total_images"], stats["images_with_detections"],
                  stats["images_no_detection"], stats["total_boxes"]]
    })
    summary_path = Path(output_dir) / "dataset_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSummary saved to {summary_path}")


def draw_boxes_on_image(image_path, label_path):
    """Draw bounding boxes with class colours on an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    h, w = img.shape[:2]

    if not label_path.exists():
        return img

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            x_c, y_c, box_w, box_h = map(float, parts[1:5])

            x1 = int((x_c - box_w/2) * w)
            y1 = int((y_c - box_h/2) * h)
            x2 = int((x_c + box_w/2) * w)
            y2 = int((y_c + box_h/2) * h)

            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            label = ID_TO_SPECIES.get(cls_id, f"class{cls_id}")
            color = CLASS_COLORS.get(cls_id, (255, 255, 255))
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
    return img


def save_special_images(image_dir, label_dir, df, save_dir):
    """
    Find images with multiple boxes or no boxes and save them (with annotations).
    """
    image_dir = Path(image_dir)
    label_dir = Path(label_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_files = [f for f in image_dir.iterdir() if f.suffix.lower() in img_exts]
    image_files = sorted(image_files)

    multi_boxes = []
    no_boxes = []
    for img_path in image_files:
        label_path = label_dir / (img_path.stem + ".txt")
        if label_path.exists():
            with open(label_path, 'r') as f:
                box_count = len([line for line in f if line.strip()])
        else:
            box_count = 0

        if box_count == 0:
            no_boxes.append((img_path, label_path))
        elif box_count > 1:
            multi_boxes.append((img_path, label_path, box_count))

    print(f"\n--- Special images found ---")
    print(f"Images with multiple boxes: {len(multi_boxes)}")
    print(f"Images with no boxes:       {len(no_boxes)}")

    # Save multi-box images
    if multi_boxes:
        multi_save_dir = save_dir / "multiple_boxes"
        multi_save_dir.mkdir(exist_ok=True)
        print(f"\nSaving {len(multi_boxes)} multi‑box images to {multi_save_dir}")
        for img_path, label_path, count in multi_boxes:
            annotated = draw_boxes_on_image(img_path, label_path)
            if annotated is not None:
                out_path = multi_save_dir / img_path.name
                cv2.imwrite(str(out_path), annotated)

    # Save no-box images
    if no_boxes:
        no_save_dir = save_dir / "no_boxes"
        no_save_dir.mkdir(exist_ok=True)
        print(f"Saving {len(no_boxes)} no‑box images to {no_save_dir}")
        for img_path, label_path in no_boxes:
            annotated = draw_boxes_on_image(img_path, label_path)  # plain image
            if annotated is not None:
                out_path = no_save_dir / img_path.name
                cv2.imwrite(str(out_path), annotated)

    print("All special images saved.")


def create_data_yaml(output_dir, class_names):
    yaml_content = f"""
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

    copy_images(df, IMAGE_SOURCE_DIR, OUTPUT_DIR)
    print("Images copied.")

    print("Running YOLO detection and enforcing CSV labels...")
    stats = enforce_labels_with_csv(MODEL_PATH, IMG_DIR, LABEL_DIR, df, CONF_THRESHOLD)
    print("Label enforcement complete.")

    summarize_dataset(stats, OUTPUT_DIR)

    class_names = list(SPECIES_TO_ID.keys())
    create_data_yaml(OUTPUT_DIR, class_names)
    print(f"Dataset prepared in {OUTPUT_DIR}")

    if SAVE_SPECIAL_IMAGES:
        save_special_images(IMG_DIR, LABEL_DIR, df, SPECIAL_IMAGES_DIR)


if __name__ == "__main__":
    main()
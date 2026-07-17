import cv2
from pathlib import Path

CLASS_NAMES = {
    0: "anopheles_gambiae",
    1: "anopheles_stephensi",
    2: "aedes_aegypti"
}

# Define BGR colors for each class (green, red, blue, etc.)
CLASS_COLORS = {
    0: (0, 255, 0),    # green
    1: (0, 0, 255),    # red
    2: (255, 255, 0),  # cyan
}

def draw_boxes(image_path, label_path, class_names=CLASS_NAMES, class_colors=CLASS_COLORS):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not read: {image_path}")
        return None

    h, w = img.shape[:2]

    if not Path(label_path).exists():
        print(f"No label file: {label_path}")
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

            label = class_names.get(cls_id, f"class_{cls_id}")
            color = class_colors.get(cls_id, (255, 255, 255))  # default white

            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
    return img

def save_annotated_images(image_folder, label_folder, num_images, output_folder):
    """
    Process the first `num_images` images, draw bounding boxes, and save them to output_folder.
    """
    image_folder = Path(image_folder)
    label_folder = Path(label_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not image_folder.exists():
        print(f"Image folder not found: {image_folder}")
        return

    img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    img_files = [f for f in image_folder.iterdir() if f.suffix.lower() in img_exts]
    img_files = sorted(img_files)

    if not img_files:
        print("No image files found.")
        return

    num_to_process = min(num_images, len(img_files))
    print(f"Processing {num_to_process} images out of {len(img_files)}...")

    for i, img_path in enumerate(img_files[:num_to_process]):
        label_path = label_folder / (img_path.stem + '.txt')
        annotated = draw_boxes(img_path, label_path)
        if annotated is None:
            continue

        out_path = output_folder / img_path.name
        cv2.imwrite(str(out_path), annotated)
        print(f"Saved: {out_path}")

    print(f"All annotated images saved to {output_folder}")

# ------------------- CONFIGURE HERE -------------------
if __name__ == "__main__":
    IMAGE_FOLDER = "./yolo_dataset/images"
    LABEL_FOLDER = "./yolo_dataset/labels"
    OUTPUT_FOLDER = "./annotated_images"
    NUMBER_TO_PLOT = 5

    save_annotated_images(IMAGE_FOLDER, LABEL_FOLDER, NUMBER_TO_PLOT, OUTPUT_FOLDER)
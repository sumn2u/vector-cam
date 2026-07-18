## Application Setup
The folder contains the source files used to create and explore models for malaria detection. Here, we can find the placeholder directories (`0610`, `0618`, `0623`and `kenya_01/`) of images. The original images needs to be downloaded from the [google drive](https://drive.google.com/drive/folders/1T4W8j-NyJviCIUhu_3hobbugmrw_wu7M) and placed inside respective folder.

The folder strucure should look like this.
```sh
vector-cam/
├── kenya_01/
│   ├── image_13133_78265aff136f61e9dc4d3bd9ef5c98f4.jpg
│   └── ...
└── 0610/
    ├── image_26920_7bfbafc333b0a313fa54576d66eb8534.jpg
    └── ...
    ....
```
### Folder Structure.

- **Date directories** (`0610`, `0618`, `0623`and `kenya_01/`):  Contain image files of specimen. 
- **`csvs/`**: Contains CSV files of the datasets and its associate metadata.
- **Scripts**:
  - `prepare_yolo_dataset.py`: Parses CSV annotations and generates YOLO‑compatible label files (`.txt`) with normalized coordinates.
  - `enforce_labels_from_csv.py`: Ensures that the label files are consistent with the CSV ground truth, correcting any discrepancies.
  - `visualize_samples.py`: Renders images with bounding boxes overlaid for quick inspection of the dataset.
- **`kenya-experiments.ipynb`**: Jupyter notebook used for exploring the data, training YOLO models, and evaluating performance.
- **`requirements.txt`**: Lists all Python packages required to run the scripts (e.g., `opencv-python`, `pandas`, `numpy`, `matplotlib`, `ultralytics`).
 **`best.pt`**: YOLO model file obtained from [Kenya_01](https://www.kaggle.com/code/sumn2u/kenya-experiments) experiments. 

---

## Development Setup

1. **Clone the repository**  
   ```bash
   git clone <repository-url>
   cd <repository-folder>
    ```


### Running Application
1. Navigate to the `root` directory:
   ```sh
   cd vector-cam
    ```
2. Create and activate a virtual environment:
    ```sh
   python3 -m venv venv

   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install the dependencies:
```sh
   pip install -r requirements.txt
```
4. Run the associate files:
```sh
   python3 <file>
```

### Running Jupyter Notebook

A Kaggle version of the Jupyter notebook is also available to replicate the same behavior:

[https://www.kaggle.com/code/sumn2u/kenya-experiments](https://www.kaggle.com/code/sumn2u/kenya-experiments)

### Issue while Running?
Report an issue to <suman@dwaste.live>.

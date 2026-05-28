# Bark Imagery Notebook Guide

This repository contains a Jupyter notebook, `bark_notebook.ipynb`, for training image classification models on tree bark images. The notebook loads a tree-level labels CSV, prepares bark image paths, filters species, builds training/validation/test splits, and trains PyTorch/TorchVision models through the `TrainTestBarkModel` class.

The main workflow is:

1. Set up the project folder and image data.
2. Create and activate a local Python virtual environment named `.venv`.
3. Install the Python packages used by the notebook.
4. Open `bark_notebook.ipynb` in VS Code.
5. Select the `.venv` kernel.
6. Run the notebook cells from top to bottom.
7. Choose a training option at the bottom of the notebook.

## Expected Project Structure

Open VS Code at the repository root. The repository root should be the folder that contains `bark_notebook.ipynb` and `README.md`.

The local project should look like this:

```text
Bark_Imagery/
|-- README.md
|-- bark_notebook.ipynb
|-- BarkImageryPlayground.ipynb
|-- .gitignore
|-- .gitattributes
|-- .venv/
`-- bark_images_and_labels/
    |-- bark_class_labels.csv
    |-- BARK_UGA_PICS/
    |   |-- image_file_1.jpg
    |   |-- image_file_2.jpg
    |   `-- ...
    |-- BARK_UGA_PICS.zip
    |-- Related Literature/
    |-- SpatialData_100125/
    `-- Web App Requirements - Machine Learning Classifier.docx
```

Only the files required by the notebook are:

```text
bark_notebook.ipynb
bark_images_and_labels/bark_class_labels.csv
bark_images_and_labels/BARK_UGA_PICS/
```

The notebook expects `bark_class_labels.csv` to contain one row per tree and columns similar to:

```text
OBJECTID
Species
Photo0
Photo1
Photo2
Photo3
```

Each `Photo0` through `Photo3` value should be an image filename that exists inside:

```text
bark_images_and_labels/BARK_UGA_PICS/
```

For example, if the CSV contains:

```text
Photo0 = IMG_1234.jpg
```

then this file should exist:

```text
bark_images_and_labels/BARK_UGA_PICS/IMG_1234.jpg
```

The notebook builds full image paths by joining:

```python
extract_path = "bark_images_and_labels/BARK_UGA_PICS/"
```

with the image filename from the CSV.

## Data Setup

### If You Already Have the Extracted Folder

If `bark_images_and_labels/BARK_UGA_PICS/` already exists and contains the bark image files, no extra extraction step is needed.

Check that the key files are present:

```powershell
Get-ChildItem bark_images_and_labels
Get-ChildItem bark_images_and_labels\BARK_UGA_PICS
```

You should see `bark_class_labels.csv` in `bark_images_and_labels/` and many image files inside `BARK_UGA_PICS/`.

### If You Only Have the Zip File

If you have `bark_images_and_labels.zip` or `bark_images_and_labels/BARK_UGA_PICS.zip`, extract it before running the notebook.

The final folder must be:

```text
bark_images_and_labels/BARK_UGA_PICS/
```

Avoid creating an extra nested folder like this:

```text
bark_images_and_labels/BARK_UGA_PICS/BARK_UGA_PICS/
```

If the images are nested one folder too deep, move the image files up so that `BARK_UGA_PICS/` directly contains the image files.

## Virtual Environment Setup

The `.venv` folder is the local Python environment for this project. Keeping dependencies in `.venv` prevents package conflicts with other Python projects on your computer.

Run these commands from the repository root in the VS Code terminal.

### Windows PowerShell

Create the virtual environment:

```powershell
py -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation scripts, run this once:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Then activate the environment again:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade `pip`:

```powershell
python -m pip install --upgrade pip
```

Install the packages used by the notebook:

```powershell
python -m pip install pandas numpy scikit-learn pillow torch torchvision jupyter ipykernel matplotlib
```

Register the virtual environment as a Jupyter kernel:

```powershell
python -m ipykernel install --user --name bark-imagery --display-name "Python (.venv: Bark Imagery)"
```

### macOS or Linux

Create the virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Upgrade `pip`:

```bash
python -m pip install --upgrade pip
```

Install the packages used by the notebook:

```bash
python -m pip install pandas numpy scikit-learn pillow torch torchvision jupyter ipykernel matplotlib
```

Register the virtual environment as a Jupyter kernel:

```bash
python -m ipykernel install --user --name bark-imagery --display-name "Python (.venv: Bark Imagery)"
```

## Opening the Notebook in VS Code

1. Open VS Code.
2. Open the repository folder, not just the notebook file.
3. Install the VS Code Python extension if it is not already installed.
4. Install the VS Code Jupyter extension if it is not already installed.
5. Open `bark_notebook.ipynb`.
6. Click the kernel selector in the top-right corner of the notebook.
7. Select `Python (.venv: Bark Imagery)`.
8. Run the notebook cells from top to bottom.

Running from the repository root matters because the notebook uses relative paths such as:

```python
labels_path = "bark_images_and_labels/bark_class_labels.csv"
extract_path = "bark_images_and_labels/BARK_UGA_PICS/"
```

If the notebook is launched from a different working directory, those paths may not resolve correctly.

## Running in Google Colab

The notebook also contains a small Colab path check:

```python
IN_COLAB = os.path.exists("/content")
```

If the notebook detects Colab, it mounts Google Drive and sets:

```python
BASE_PATH = "/content/drive/MyDrive/Bark_Imagery/"
```

For Colab, place the project data in:

```text
MyDrive/Bark_Imagery/
```

with the same internal structure:

```text
MyDrive/Bark_Imagery/
|-- bark_notebook.ipynb
`-- bark_images_and_labels/
    |-- bark_class_labels.csv
    `-- BARK_UGA_PICS/
```

If you use a different Google Drive folder name, update `BASE_PATH` in the first code cell.

## Notebook Walkthrough

The notebook is organized into several major sections.

### 1. Load Labels and Define Paths

The first code section imports:

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import os
```

It then defines:

```python
labels_path = BASE_PATH + "bark_images_and_labels/bark_class_labels.csv"
extract_path = BASE_PATH + "bark_images_and_labels/BARK_UGA_PICS/"
```

The labels CSV is loaded into a pandas dataframe:

```python
df = pd.read_csv(labels_path)
```

Rows where `Species` is `"other"` are removed:

```python
df = df[~df['Species'].str.lower().eq('other')].copy()
```

This cleaned `df` is the main tree-level dataframe used later in the notebook.

### 2. Create a Small Sample Split

The notebook creates a small stratified sample:

```python
sample_tree_df, remaining_tree_df = train_test_split(
    df,
    train_size=0.05,
    random_state=42,
    stratify=df["Species"]
)
```

This keeps the initial training run small and fast. The split is stratified, which means the species distribution is preserved as much as possible.

Use this sample when you want to confirm that the training loop works before running bigger experiments.

### 3. Count Species

The notebook creates a species count table:

```python
species_counts_df = (
    df["Species"]
    .value_counts()
    .rename_axis("Species")
    .reset_index(name="TreeCount")
)
```

This helps identify rare species that may not have enough trees for stratified training, validation, testing, or five-fold cross-validation.

### 4. Melt Tree Rows into Image Rows

The raw CSV is tree-level data: one row represents one tree, and each tree can have multiple photo columns:

```text
Photo0
Photo1
Photo2
Photo3
```

The notebook converts this tree-level format into image-level format using `melt_photos`.

Tree-level format:

```text
OBJECTID | Species | Photo0      | Photo1      | Photo2      | Photo3
598      | White Oak | img_a.jpg | img_b.jpg   | img_c.jpg   | img_d.jpg
```

Image-level format:

```text
OBJECTID | Species   | Photo  | Image
598      | White Oak | Photo0 | img_a.jpg
598      | White Oak | Photo1 | img_b.jpg
598      | White Oak | Photo2 | img_c.jpg
598      | White Oak | Photo3 | img_d.jpg
```

The notebook then adds `ImagePath`:

```python
image_df["ImagePath"] = extract_path + image_df["Image"]
```

After this step, each image row has the path needed by PyTorch to load the actual image file.

### 5. Define `BarkDataset`

`BarkDataset` is a PyTorch `Dataset`.

It accepts an image-level dataframe with at least:

```text
ImagePath
Label
```

For each row, it:

1. Reads the image from `ImagePath`.
2. Converts the image to RGB.
3. Applies the transform passed into the dataset.
4. Converts the numeric label into a PyTorch tensor.
5. Returns `(image, label)`.

The important method is:

```python
def __getitem__(self, index):
    row = self.image_df.iloc[index]
    image_path = row["ImagePath"]
    label = row["Label"]
    image = Image.open(image_path).convert("RGB")
    image = self.transform(image)
    label = torch.tensor(label, dtype=torch.long)
    return image, label
```

You usually do not call `BarkDataset` directly. `TrainTestBarkModel.get_dataloader()` creates it for you.

## `TrainTestBarkModel` Overview

`TrainTestBarkModel` is the main helper class in `bark_notebook.ipynb`.

It handles:

- Label encoding.
- Tree-level train/validation/test splitting.
- Optional five-fold cross-validation.
- Melting tree-level rows into image-level rows.
- Building image paths.
- Creating PyTorch dataloaders.
- Creating image transforms.
- Training one model.
- Training multiple models.
- Evaluating validation and test accuracy.
- Returning results as pandas dataframes.

The class is designed around two input modes:

1. Image dataframe mode: pass `train_image_df`.
2. Tree dataframe mode: pass `tree_df`.

You must use one mode or the other. Do not pass both.

## `TrainTestBarkModel` Constructor

The constructor is:

```python
TrainTestBarkModel(
    train_image_df=None,
    val_image_df=None,
    test_image_df=None,
    tree_df=None,
    melt_labels=["Photo0", "Photo1", "Photo2", "Photo3"],
    extract_path="bark_images_and_labels/BARK_UGA_PICS/",
    five_fold_cross_validation=False,
    data_fraction=1.0,
    train_size=0.7,
    val_size=0.1,
    test_size=0.2,
    random_state=42
)
```

### Constructor Parameters

| Parameter | Purpose |
| --- | --- |
| `train_image_df` | An already-melted image-level dataframe used for training. Must contain `Species` and `ImagePath`. |
| `val_image_df` | Optional image-level validation dataframe. Only use this with `train_image_df` mode. |
| `test_image_df` | Optional image-level test dataframe. Only use this with `train_image_df` mode. |
| `tree_df` | A tree-level dataframe with `OBJECTID`, `Species`, and photo columns. Use this when you want the class to create splits for you. |
| `melt_labels` | The photo columns to melt into image rows. Defaults to `Photo0` through `Photo3`. |
| `extract_path` | Folder containing the image files. Used to create `ImagePath` when `tree_df` mode is used. |
| `five_fold_cross_validation` | If `True`, creates five stratified folds instead of one train/validation/test split. |
| `data_fraction` | Fraction of the tree dataframe to use. Helpful for smaller experiments. |
| `train_size` | Fraction of selected tree data used for training. Default is `0.7`. |
| `val_size` | Fraction of selected tree data used for validation. Default is `0.1`. |
| `test_size` | Fraction of selected tree data used for testing. Default is `0.2`. |
| `random_state` | Random seed used by the stratified splits. |

## Input Mode 1: Image Dataframe Mode

Use image dataframe mode when you already have an image-level dataframe.

The dataframe must already contain:

```text
OBJECTID
Species
Photo
Image
ImagePath
```

The class will add:

```text
Label
```

Example from the notebook:

```python
bulk_trainer = TrainTestBarkModel(
    train_image_df=sample_image_df
)
```

The notebook also uses the equivalent positional form:

```python
bulk_trainer = TrainTestBarkModel(sample_image_df, extract_path=extract_path)
```

In this mode, `extract_path` is not used to build paths. The paths must already exist in the `ImagePath` column.

This mode is best for:

- Quick training tests.
- Small sampled datasets.
- Debugging the model loop.
- Manually controlled splits.

Important limitation: the label encoder is fit on `train_image_df`. If you provide `val_image_df` or `test_image_df`, every species in those dataframes must also appear in `train_image_df`.

## Input Mode 2: Tree Dataframe Mode

Use tree dataframe mode when you want the class to handle splitting, label encoding, melting, and path creation.

The dataframe must contain:

```text
OBJECTID
Species
Photo0
Photo1
Photo2
Photo3
```

Example:

```python
bulk_trainer_and_eval = TrainTestBarkModel(
    tree_df=df_filtered_species,
    extract_path=extract_path,
    data_fraction=0.2
)
```

In this mode, the class:

1. Resets the dataframe index.
2. Stores the photo columns from `melt_labels`.
3. Stores `extract_path`.
4. Stores the split settings.
5. Fits one label encoder on the full tree dataframe.
6. Creates train/validation/test tree splits.
7. Adds numeric labels to each split.
8. Melts each tree split into image-level rows.
9. Builds `ImagePath` for each image row.
10. Stores the final image-level splits as:

```python
trainer.train_image_df
trainer.val_image_df
trainer.test_image_df
```

This mode is best for:

- Full experiments.
- Validation and test evaluation.
- Avoiding image leakage across splits.
- Keeping all photos from the same tree in the same split.

Keeping all photos from the same tree together matters. If photos from the same tree appear in both training and testing, the evaluation can look better than it really is because the model has already seen nearly identical bark from the same tree.

## Five-Fold Cross-Validation Mode

Use five-fold cross-validation when you want every eligible tree to appear in a test fold exactly once.

Example:

```python
bulk_trainer_and_cross_val = TrainTestBarkModel(
    tree_df=df_filtered_species,
    extract_path=extract_path,
    five_fold_cross_validation=True
)
```

With the default split values:

```python
train_size = 0.7
val_size = 0.1
test_size = 0.2
```

five-fold cross-validation creates five folds. In each fold:

- 20% of trees are held out for testing.
- The remaining 80% are split into training and validation.
- The model is trained and evaluated for that fold.

The class stores these folds in:

```python
trainer.cv_folds
```

Each fold dictionary contains:

```text
fold
train_tree_df
val_tree_df
test_tree_df
train_image_df
val_image_df
test_image_df
```

You can manually select a fold with:

```python
trainer.set_fold(0)
```

Fold indexes are zero-based, so:

```text
0 = fold 1
1 = fold 2
2 = fold 3
3 = fold 4
4 = fold 5
```

Usually, you do not need to call `set_fold()` manually. `train_model()` and `train_multiple()` automatically iterate over all folds when `five_fold_cross_validation=True`.

Five-fold constraints:

- `data_fraction` must be `1.0`.
- `test_size` must be `0.2`.
- `train_size + val_size` must equal `0.8`.
- Each species must have enough trees for stratified fold creation and validation splitting.

The notebook filters species before cross-validation:

```python
min_trees_per_species = 34
species_counts = df["Species"].value_counts()
kept_species = species_counts[species_counts >= min_trees_per_species].index
df_filtered_species = df[df["Species"].isin(kept_species)].copy()
```

This removes species with too few trees for reliable stratified splitting.

## Label Encoding

The class converts species names into numeric labels with scikit-learn's `LabelEncoder`.

For example:

```text
American Beech -> 0
Black Oak      -> 1
White Oak      -> 2
```

The actual numbers depend on the sorted species names in the dataframe.

The label information is stored as:

```python
trainer.label_encoder
trainer.num_classes
trainer.label_map_df
```

To display the mapping:

```python
trainer.show_label_map()
```

To get the number of classes:

```python
num_classes = trainer.get_num_classes()
```

The final model layer is replaced so that the model outputs exactly `num_classes` logits.

## Important Methods

### `set_label_info(label_fit_df)`

Fits the label encoder on the `Species` column and creates:

```python
self.num_classes
self.label_map_df
self.label_encoder
```

You normally do not call this directly. It is called inside the constructor.

### `split_data(tree_df)`

Creates one stratified train/validation/test split from a tree-level dataframe.

It validates:

- `0.0 < data_fraction <= 1.0`
- `train_size + val_size + test_size == 1.0`
- each species has enough trees for the requested split

Then it:

1. Optionally samples `data_fraction` of the tree dataframe.
2. Splits the selected trees into train and temporary data.
3. Splits the temporary data into validation and test data.
4. Builds image-level train/validation/test dataframes.

### `split_data_five_fold(tree_df)`

Creates five stratified folds from a tree-level dataframe.

It validates the cross-validation settings, then creates five fold dictionaries containing both tree-level and image-level splits.

### `build_image_splits(train_tree_df, val_tree_df, test_tree_df)`

Applies label encoding, melts photo columns, and builds image paths for each split.

Returns:

```python
train_image_df, val_image_df, test_image_df
```

### `melt_photos(split_df)`

Converts tree-level rows into image-level rows.

Input columns:

```text
OBJECTID
Species
Label
Photo0
Photo1
Photo2
Photo3
```

Output columns:

```text
OBJECTID
Species
Label
Photo
Image
```

Rows with missing image filenames are dropped.

### `build_image_paths(image_df)`

Adds the full image path:

```python
image_df["ImagePath"] = self.extract_path + image_df["Image"]
```

### `validate_min_species_count(...)`

Checks whether each species has enough trees for the requested stratified split.

If not, it raises a `ValueError` showing which species are too small.

Common fixes are:

- Increase `data_fraction`.
- Lower `train_size` so more data remains for validation/test.
- Remove rare species with a higher `min_trees_per_species` filter.
- Use fewer species for the experiment.

### `get_transform(weights=None, image_size=None)`

Returns the image preprocessing transform.

If TorchVision pretrained weights are provided and `image_size` is `None`, it uses the transform that belongs to those weights:

```python
weights.transforms()
```

If no weights transform is used, it creates this default transform:

```python
transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
])
```

If `image_size` is also `None`, the default image size is:

```python
(224, 224)
```

### `get_dataloader(batch_size, shuffle, image_df=None, transform=None)`

Creates a PyTorch `DataLoader`.

Example:

```python
transform = trainer.get_transform(image_size=(224, 224))
dataloader = trainer.get_dataloader(
    batch_size=32,
    shuffle=True,
    image_df=trainer.train_image_df,
    transform=transform
)
```

`transform` is required. If it is missing, the method raises a `ValueError`.

### `train_model(model_tuple, batch_size, num_epochs)`

Trains one model.

The `model_tuple` format is:

```python
(model_name, model_function, weights, image_size)
```

Example:

```python
("resnet18", resnet18, ResNet18_Weights.DEFAULT, None)
```

or:

```python
("resnet18", resnet18, None, (224, 224))
```

Inside `train_model`, the class:

1. Chooses `cuda` if available, otherwise `cpu`.
2. Creates the model using `model_function(weights=weights)`.
3. Builds train and evaluation transforms.
4. Replaces the model's final classification layer.
5. Uses `CrossEntropyLoss`.
6. Uses the Adam optimizer with learning rate `0.001`.
7. Trains for `num_epochs`.
8. Evaluates on validation data if available.
9. Evaluates on test data if available.
10. Returns a results dataframe.

If `num_epochs` is `None`, the notebook sets:

```python
current_num_epochs = number_of_training_images // batch_size
```

For example, if there are 3,200 training images and `batch_size=32`, the model trains for:

```text
3200 // 32 = 100 epochs
```

For a quick test, set `NUM_EPOCHS = 1`.

### `evaluate_model(model, image_df, transform, batch_size, split_name)`

Evaluates a trained model on a validation or test dataframe.

It returns:

```python
avg_loss, accuracy
```

It also prints:

```text
Validation Loss
Validation Accuracy
```

or:

```text
Test Loss
Test Accuracy
```

depending on `split_name`.

### `train_multiple(models, batch_size, num_epochs)`

Trains every model tuple in a list.

Example:

```python
results_df = trainer.train_multiple(
    models=models_default_weights_partial,
    batch_size=32,
    num_epochs=1
)
```

If one model fails, the method records the failure and continues to the next model. It returns one combined results dataframe.

## Model Lists in the Notebook

The notebook imports ResNet and EfficientNet models from TorchVision.

The available model lists are:

| Model list | Description |
| --- | --- |
| `models_default_weights_full` | Full ResNet, EfficientNet B, and EfficientNet V2 list using default pretrained weights and default weight transforms. |
| `models_default_weights_partial` | Smaller model list using default pretrained weights. Better for faster experiments. |
| `models_default_weights_custom_imagesize_full` | Full model list using default pretrained weights but manually specified image sizes. |
| `models_default_weights_custom_imagesize_partial` | Smaller custom-image-size model list. |

Each model tuple has this structure:

```python
("model_name", model_constructor, weights, image_size)
```

Examples:

```python
("resnet18", resnet18, ResNet18_Weights.DEFAULT, None)
("efficientnet_b1", efficientnet_b1, EfficientNet_B1_Weights.DEFAULT, (240, 240))
```

When `weights` is not `None`, TorchVision may download pretrained weights the first time you run that model. If you do not have internet access, use `weights=None` for that model.

## Training Options at the Bottom of the Notebook

The final cell uses `MODEL_OPTION` to choose what to run.

```python
MODEL_OPTION = 7
BATCH_SIZE = 32
NUM_EPOCHS = None
```

Options:

| `MODEL_OPTION` | What it runs |
| --- | --- |
| `0` | Single `resnet18` training run on the small sampled image dataframe. Good first test. |
| `1` | Full model list training without validation/test evaluation. |
| `2` | Full model list with validation/test evaluation using default weights. |
| `3` | Partial model list with validation/test evaluation using default weights. |
| `4` | Full model list with validation/test evaluation and custom image sizes. |
| `5` | Partial model list with validation/test evaluation and custom image sizes. |
| `6` | Five-fold cross-validation on the full default-weight model list. |
| `7` | Five-fold cross-validation on the partial default-weight model list. |
| `8` | Five-fold cross-validation on the full custom-image-size model list. |
| `9` | Five-fold cross-validation on the partial custom-image-size model list. |

For your first run, use:

```python
MODEL_OPTION = 0
BATCH_SIZE = 32
NUM_EPOCHS = 1
```

This checks that the environment, image paths, dataloaders, and model training loop work.

After that, try:

```python
MODEL_OPTION = 3
BATCH_SIZE = 32
NUM_EPOCHS = 1
```

Then increase `NUM_EPOCHS` when you are ready for longer training.

## Recommended First Successful Run

Use this sequence when setting up the project for the first time.

1. Confirm the data folder exists:

```text
bark_images_and_labels/BARK_UGA_PICS/
```

2. Confirm the labels file exists:

```text
bark_images_and_labels/bark_class_labels.csv
```

3. Create and activate `.venv`.

4. Install dependencies.

5. Open `bark_notebook.ipynb`.

6. Select the `.venv` kernel.

7. Run the first cells through the image path verification cell.

8. Look for this check:

```python
path_exists_df = sample_image_df["ImagePath"].apply(os.path.exists)
print(path_exists_df.value_counts())
```

You want the output to show mostly or entirely:

```text
True
```

If it shows `False`, fix the folder structure or `extract_path` before training.

9. Set:

```python
MODEL_OPTION = 0
BATCH_SIZE = 32
NUM_EPOCHS = 1
```

10. Run the final cell.

If that works, the project is set up correctly.

## Common Usage Examples

### Example 1: Train One ResNet18 on the Small Sample

```python
bulk_trainer = TrainTestBarkModel(
    train_image_df=sample_image_df
)

results_df = bulk_trainer.train_model(
    model_tuple=("resnet18", resnet18, None, (224, 224)),
    batch_size=32,
    num_epochs=1
)
```

This is the fastest way to test the training loop.

### Example 2: Train and Evaluate a Partial Model List

```python
bulk_trainer_and_eval = TrainTestBarkModel(
    tree_df=df_filtered_species,
    extract_path=extract_path,
    data_fraction=0.2
)

results_df = bulk_trainer_and_eval.train_multiple(
    models=models_default_weights_partial,
    batch_size=32,
    num_epochs=1
)
```

This creates train/validation/test splits and evaluates each model.

### Example 3: Run Five-Fold Cross-Validation

```python
bulk_trainer_and_cross_val = TrainTestBarkModel(
    tree_df=df_filtered_species,
    extract_path=extract_path,
    five_fold_cross_validation=True
)

results_df = bulk_trainer_and_cross_val.train_multiple(
    models=models_default_weights_partial,
    batch_size=32,
    num_epochs=1
)
```

This trains each model once per fold and returns results by fold.

### Example 4: View the Label Map

```python
bulk_trainer_and_eval.show_label_map()
```

This displays the numeric label assigned to each species.

### Example 5: Check Split Sizes

```python
print("Train images:", len(bulk_trainer_and_eval.train_image_df))
print("Validation images:", len(bulk_trainer_and_eval.val_image_df))
print("Test images:", len(bulk_trainer_and_eval.test_image_df))
```

### Example 6: Check Whether Image Paths Exist

```python
path_exists = bulk_trainer_and_eval.train_image_df["ImagePath"].apply(os.path.exists)
print(path_exists.value_counts())
```

If you see `False`, the image folder path is wrong or the image files are missing.

## Performance Notes

Training image classifiers can be slow, especially on CPU.

To make experiments faster:

- Start with `MODEL_OPTION = 0`.
- Set `NUM_EPOCHS = 1`.
- Use `models_default_weights_partial`.
- Use a smaller `data_fraction`, such as `0.2`.
- Lower `BATCH_SIZE` if memory runs out.
- Avoid EfficientNet B5, B6, B7, and large EfficientNet V2 models until smaller models work.

If your computer has an NVIDIA GPU and a CUDA-enabled PyTorch install, the notebook will automatically use it:

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

If CUDA is not available, it will use CPU.

## Troubleshooting

### `FileNotFoundError` or `Error loading image at ...`

The image path is probably wrong.

Check:

```python
print(extract_path)
print(sample_image_df["ImagePath"].head())
print(sample_image_df["ImagePath"].apply(os.path.exists).value_counts())
```

Fixes:

- Make sure VS Code opened the repository root folder.
- Make sure `bark_images_and_labels/BARK_UGA_PICS/` exists.
- Make sure the image files are directly inside `BARK_UGA_PICS/`.
- Make sure the filenames in `bark_class_labels.csv` match the actual image filenames.

### `ValueError: Some species do not have enough trees...`

The requested stratified split needs more examples per species.

Fixes:

- Increase `data_fraction`.
- Use the full dataframe with `data_fraction=1.0`.
- Increase `min_trees_per_species`.
- Remove rare species.
- Use a simpler train/test setup before trying five-fold cross-validation.

### `ValueError: Train, validation, and test sizes must sum to 1.0`

Make sure:

```python
train_size + val_size + test_size == 1.0
```

For example:

```python
train_size=0.7
val_size=0.1
test_size=0.2
```

### Five-Fold Cross-Validation Errors

For five-fold cross-validation, these must be true:

```python
data_fraction == 1.0
test_size == 0.2
train_size + val_size == 0.8
```

The default values already satisfy this:

```python
train_size=0.7
val_size=0.1
test_size=0.2
```

You also need enough trees per species.

### `Unsupported model architecture`

The training code currently supports models with either:

```python
model.fc
```

or a sequential classifier:

```python
model.classifier[-1]
```

If you add a model with a different classifier structure, you will need to update the final-layer replacement logic in `train_model`.

### Out-of-Memory Errors

Fixes:

- Lower `BATCH_SIZE`.
- Use `models_default_weights_partial`.
- Use fewer epochs.
- Use smaller models first, such as `resnet18`.
- Restart the notebook kernel between large experiments.

Example:

```python
BATCH_SIZE = 8
NUM_EPOCHS = 1
MODEL_OPTION = 0
```

### Pretrained Weights Will Not Download

TorchVision downloads pretrained weights the first time a pretrained model is used.

Fixes:

- Connect to the internet and rerun the cell.
- Use `weights=None` in the model tuple.

Example:

```python
("resnet18", resnet18, None, (224, 224))
```

## Suggested Workflow for Experiments

Use this progression:

1. Run `MODEL_OPTION = 0`, `NUM_EPOCHS = 1`.
2. Confirm images load and the training loop completes.
3. Run `MODEL_OPTION = 3`, `NUM_EPOCHS = 1`.
4. Compare validation/test accuracy across the partial model list.
5. Increase `NUM_EPOCHS`.
6. Try cross-validation with `MODEL_OPTION = 7`.
7. Try larger model lists only after the smaller runs are stable.

This avoids spending a long time on large experiments before confirming that the data paths, labels, splits, and training loop are working correctly.

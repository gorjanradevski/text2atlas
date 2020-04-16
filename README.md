# Self-supervised context-aware Covid-19 document exploration through atlas grounding

This repository is the official implementation of [Self-supervised context-aware Covid-19 document exploration through atlas grounding](https://github.com/gorjanradevski/macchina) authored by Dusan Grujicic<sup>*</sup>, [Gorjan Radevski<sup>*</sup>](http://gorjanradevski.github.io/), [Tinne Tuytelaars](https://homes.esat.kuleuven.be/~tuytelaa/), [Matthew Blaschko](https://homes.esat.kuleuven.be/~mblaschk/). The work is currently under review at [NLP COVID-19 Workshop](https://www.nlpcovid19workshop.org/) at ACL 2020.

See our [Cord-19 Explorer](https://cord19-explorer.herokuapp.com/) and our [Cord-19 Visualizer](https://github.com/dusangrujicic/cord19-visualizer) tools.

><sup>*</sup> Equal contribution

## Requirements

If you are using [Poetry](https://python-poetry.org/), navigating to the project root directory and running `poetry install` will suffice. Otherwise, a `requirements.txt` file is present at the project root directory so you can install all dependencies by running `pip install -r requirements.txt`. However, if you just want to download the trained models or dataset splits, make sure to have [gdown](https://github.com/wkentaro/gdown) installed. If the project dependencies are installed then `gdown` is already present, otherwise, run `pip install gdown` to install it.

## Fetching the data

The data we use to perform the research consist of the splits used for training, validation and testing the model, together with a [3D human model](https://www.voxel-man.com/segmented-inner-organs-of-the-visible-human/). 

### Downloading the dataset splits

The training, validation and test splits obtained from the [original dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge), plus the corresponding mappings to the human body atlas can be downloaded with `gdown` using the code snippet bellow.

```shell
gdown "https://drive.google.com/uc?id=1kLvbRVzyR-66lrfzLfeFd3k9-l_S_Cl4" -O data/cord_dataset_train.json
gdown "https://drive.google.com/open?id=1mnlcI5HwgY9RaCqPyWmpEeftnqIxAUQQ" -O data/cord_dataset_val.json
gdown "https://drive.google.com/uc?id=18VSbspzB2VjxDdLaVSNyFB-GZAvEopGE" -O data/cord_dataset_test.json
```

### Downloading the 3D human model

Instructions for obtaining the human atlas can be found on the [Voxel-Man website] (https://www.voxel-man.com/segmented-inner-organs-of-the-visible-human/)

The obtained model contains images of the male head (head.zip) and torso (innerorgans.zip). The unzipped directory innerograns/, contains the a text file with a of objects and their segmentation labels, and three directories, CT/, labels/, rgb/.

The innerorgans/labels/ directory constains slices of the human atlas in the form of .tif images, where the grayscale level represents the segmentation label for each organ. It is used for training and evaluating the model, and should be moved to the data/ directory in the project prior to running the scripts.

### Generating required json files

The data/data_organs_cord/ directory already contains four json files with dictionaries related to the organs in the human atlas:
  - organ2ind.json
  - ind2organ.json
  - organ2label.json
  - organ2alias.json
  
Details of the steps (removals, mergers of organ segmentation labels and renamings) that resulted in such json files can be found [here](data/README.md).
  
An additional three json files need to be generated after obtaining the human atlas and moving the labels/ directory with images to the data/ directory of the project.
This can be done by running the following script:
```shell
python src/generate_voxel_dict.py --organs_dir_path "data/data_organs_cord"\
                                  --voxelman_images_path "data/labels"
```
This script should generate three additional json files:
  - organ2voxels.json
  - organ2voxels_eroded.json
  - organ2summary.json
And place them in the data/data_organs_cord/ directory.
  

## Training

To train a new model on the training data split, from the root project directory run:

```shell
python src/train_mapping_reg.py --batch_size 128\
                                --save_model_path "models/cord_basebert_grounding.pt"\
                                --save_intermediate_model_path "models/intermediate_cord_basebert_grounding.pt"\
                                --train_json_path "data/cord_dataset_train.json"\
                                --val_json_path "data/cord_dataset_val.json"\
                                --epochs 20\
                                --bert_name "bert-base-uncased"\
                                --loss_type "all_voxels"\
                                --organs_dir_path "data/data_organs_cord"\
                                --learning_rate 2e-5
```

The script will train a model for 20 epochs, and will save the model with that reports the lowest distance to the nearest voxel on the validation set at `"models/cord_basebert_grounding.pt"`. Furthermore, keeping the arguments as they are, while changing `--bert_name` to `bert-base-uncased`, `emilyalsentzer/Bio_ClinicalBERTpytorch`, `allenai/scibert_scivocab_uncased` or `emilyalsentzer/Bio_ClinicalBERT`, will reproduce the `BertBase`, `BioBert`, `SciBert` and `ClinicalBert` models from the paper accordingly. To train the model we use for the [Cord-19 Explorer tool](https://cord19-explorer.herokuapp.com/), the `--bert_name` argument should be changed to `google/bert_uncased_L-4_H-512_A-8`, `--learning_rate` to `5e-5` and `--epochs` to `50`.

## Evaluation

To perform inference on the test data split, from the root project directory run:

```shell
python src/inference_mapping_reg.py --batch_size 128\
                                    --checkpoint_path "models/cord_basebert_grounding.pt"\
                                    --test_json_path "data/cord_dataset_test.json"\
                                    --bert_name "bert-base-uncased"\
                                    --organs_dir_path "data/data_organs_cord"
```

The script will perfrom inference with the trained model saved at `models/cord_basebert_grounding.pt`, and report the Inside Organ Ratio (IOR) and Distance to the nearst Voxel metrics on the test set.

## Pre-trained models

All models used to report the results in the paper can be downloaded with `gdown` using the code snippet bellow.

```shell
gdown "https://drive.google.com/uc?id=17_2g3kWndZI64WpGSR4EZEIK2qBzLrtI" -O models/cord_basebert_grounding.pt
gdown "https://drive.google.com/uc?id=17nUZ0Iym6q7U83kO9QowdmCzvQlp7Cce" -O models/cord_biobert_grounding.pt
gdown link_goes_here -O models/cord_scibert_grounding.pt
gdown "https://drive.google.com/uc?id=144TyLhPmPnZNH88hP4WHLzAC4So7OvFU" -O models/cord_clinicalbert_grounding.pt
gdown "https://drive.google.com/uc?id=11OHi9wETRPAHUTIH4p6BqZY3gH6NJtve" -O models/cord_smallbert_grounding.pt
```

## Reference

If you found this code useful, or use some of our resources for your work, we will appreciate if you cite our paper.

```tex
BibTeX entry should go here.
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).

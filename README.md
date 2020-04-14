# Self-supervised context-aware Covid-19 document exploration through atlas grounding

This repository is the official implementation of [Self-supervised context-aware Covid-19 document exploration through atlas grounding](https://github.com/gorjanradevski).

## Requirements

If you are using [Poetry](https://python-poetry.org/), navigating in the project root directory and running `poetry install` will suffice. Otherwise, a `requirements.txt` file is present at the project root directory so you can install all dependencies by running `pip install -r requirements.txt`.

## Fetching the data-splits

The training, validation and test splits obtained from the [original dataset](https://www.kaggle.com/allen-institute-for-ai/CORD-19-research-challenge),can be downloaded by executing the code snippet below (Assuming that all project dependencies are installed).

```shell
gdown link_goes_here -O data/dataset_covid_train.json
gdown link_goes_here -O data/dataset_covid_val.json
gdown link_goes_here -O data/dataset_covid_test.json
```

## Training

To train a new model on the training data split, from the root project directory run:

```python
python src/train_mapping_reg.py --batch_size 128\
                                --save_model_path "models/covid_basebert.pt"\
                                --save_intermediate_model_path "models/intermediate_covid_smallbert.pt"\
                                --train_json_path "data/dataset_covid_train.json"\
                                --val_json_path "data/dataset_covid_val.json"\
                                --epochs 20\
                                --bert_name "bert-base-uncased"\
                                --loss_type "all_voxels"\
                                --organs_dir_path "data/data_organs_covid"\
                                --learning_rate 2e-5
```

The script will train a model for 20 epochs, and at `"models/covid_basebert.pt"` will save the model with that reports the lowest distance to the nearest voxel on the validation set. Furthermore, keeping the arguments as they are above, while changing the `--bert_name` to `bert-base-uncased`, `emilyalsentzer/Bio_ClinicalBERTpytorch`, `allenai/scibert_scivocab_uncased` or `emilyalsentzer/Bio_ClinicalBERT`, will reproduce the `BertBase`, `BioBert`, `SciBert` and `ClinicalBert` accordingly. Finally, to train the model we use for the [Covid-19 Explorer tool](https://covid19-explorer.herokuapp.com/), the `--bert_name` argument should be changed to `google/bert_uncased_L-4_H-512_A-8` (Don't forget to change the `--save_model_path` and `--save_intermediate_model_path` to your preferred names for each trained model).

## Evaluation

To perform inference on the test data split, from the root project directory run:

```python

python src/inference_mapping_reg.py --batch_size 128\
                                    --checkpoint_path "models/covid_basebert.pt"\
                                    --test_json_path "data/dataset_covid_test.json"\
                                    --bert_name "bert-base-uncased"\
                                    --organs_dir_path "data/data_organs_covid"
```

The script will perfrom inference with the trained model saved at `models/covid-basebert.pt`, and report the Inside Organ Ratio (IOR) and Distance to the nearst Voxel metrics on the test set.

## Pre-trained models

All models used to report the results in the paper can be downloaded by running the below code snippet (Assuming that all project dependencies are installed).

```shell
gdown link_goes_here -O models/covid_basebert.pt
gdown link_goes_here -O models/covid_biobert.pt
gdown link_goes_here -O models/covid_scibert.pt
gdown link_goes_here -O models/covid_clinicalbert.pt
gdown link_goes_here -O models/covid_smallbert.pt
```

## Reference

If you found this code useful, or use some of our resources for your work, we will appreciate if you cite our paper.

```
BibTeX entry should go here.
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).

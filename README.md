# Learning to ground medical text in a 3D human atlas


<p float="left">
  <img src="https://github.com/gorjanradevski/text2atlas/blob/dusan/data/animation.gif" width="592" height="480"/>
  <img src="https://github.com/gorjanradevski/text2atlas/blob/dusan/data/image.png" width="180" height="360"/> 
</p>

___

This repository is the official implementation of [Learning to ground medical text in a 3D human atlas](https://www.aclweb.org/anthology/2020.conll-1.23/) (published at [CoNLL 2020](https://www.conll.org/2020)) authored by Dusan Grujicic<sup>*</sup>, [Gorjan Radevski<sup>*</sup>](http://gorjanradevski.github.io/), [Tinne Tuytelaars](https://homes.esat.kuleuven.be/~tuytelaa/) and [Matthew Blaschko](https://homes.esat.kuleuven.be/~mblaschk/) .

><sup>*</sup> Equal contribution

## Requirements

If you are using [Poetry](https://python-poetry.org/), navigating to the project root directory and running `poetry install` will suffice. Otherwise, a `requirements.txt` file is present so you can install all dependencies by running `pip install -r requirements.txt`. However, if you just want to download the trained models or dataset splits, make sure to have [gdown](https://github.com/wkentaro/gdown) installed. If the project dependencies are installed then `gdown` is already present. Otherwise, run `pip install gdown` to install it.

### Downloading the human atlas and preparing organ data

Instructions for obtaining the human atlas can be found on the [Voxel-Man website](https://www.voxel-man.com/segmented-inner-organs-of-the-visible-human/).

The obtained model contains images of the male head `head.zip` and torso `innerorgans.zip`. The unzipped directory `innerograns/`, contains the list of objects (organs) in the file `objectlist.txt`, and three directories, `CT/`, `labels/`, `rgb/`.

The `innerorgans/rgb/` directory, which constains slices of the human atlas in the form of `.tiff` images.

After obtaining the dataset, run:
```
poetry run python src/create_organ_dicts.py --sio_atlas_path /path/to/atlas/dir --organs_dir_path /path/to/organ_jsons/dir
```
To create dictionaries with organs info and save them in the `/path/to/organ_jsons/dir` directory, given the path to the directory where the aforementioned `objectlist.txt` and `labels/` are stored.

### Creating the dataset

After downloading the training data of the 2020 BioASQ challenge from [BioASQ](http://participants-area.bioasq.org/general_information/Task8a/), you will receive one large json file (10s of GB) that is difficult to work with, split it up into `.ndjson` files and store them in a directory of choice. 
Then run:
```
poetry run python src/allmesh_dataset.py --all_mesh_path /path/to/dir/with/BioASQ/dataset/ndjsons --dst_dset_path /filepath/under/which/dataset/is/saved --organs_dir_path /path/to/organ_jsons/dir
```
Where `all_mesh_path` is the directory with BioASQ dataset ndjsons, `organs_dir_path` is the path where the aforementioned organ jsons are stored, and `dst_dset_path` is the filepath under which the dataset is saved. This will result in a stored dataset json file, as well as the train, val and test split in the same directory.

## Reference

If you found this code useful, or use some of our resources for your work, we will appreciate if you cite our paper.

```tex
@inproceedings{Grujicic2020b,
  title={Learning to ground medical text in a {3D} human atlas},
  AUTHOR = {Grujicic, D. and G. Radevski and T. Tuytelaars and M. B. Blaschko},
  YEAR = {2020},
  booktitle= {The SIGNLL Conference on Computational Natural Language Learning},
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).

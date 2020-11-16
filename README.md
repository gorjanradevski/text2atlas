# Learning to ground medical text in a 3D human atlas


<p float="left">
  <img src="data/annimation.png" width="250" />
  <img src="data/image.gif" width="250" /> 
</p>

___

This repository is the official implementation of [Learning to ground medical text in a 3D human atlas](https://www.aclweb.org/anthology/2020.conll-1.23/) (published at [CoNLL 2020](https://www.conll.org/2020)) authored by Dusan Grujicic<sup>*</sup>, [Gorjan Radevski<sup>*</sup>](http://gorjanradevski.github.io/), [Tinne Tuytelaars](https://homes.esat.kuleuven.be/~tuytelaa/) and [Matthew Blaschko](https://homes.esat.kuleuven.be/~mblaschk/) .

><sup>*</sup> Equal contribution

## Requirements

If you are using [Poetry](https://python-poetry.org/), navigating to the project root directory and running `poetry install` will suffice. Otherwise, a `requirements.txt` file is present so you can install all dependencies by running `pip install -r requirements.txt`. However, if you just want to download the trained models or dataset splits, make sure to have [gdown](https://github.com/wkentaro/gdown) installed. If the project dependencies are installed then `gdown` is already present. Otherwise, run `pip install gdown` to install it.

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

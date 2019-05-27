# Continual Reinforcement Learning in 3D Non-stationary Environments 

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](http://creativecommons.org/licenses/by/4.0/)
[![built with Python3.x](https://img.shields.io/badge/build%20with-python3.x-red.svg)](https://www.python.org/)
[![built with PyTorch 0.3](https://img.shields.io/badge/build%20with-PyTorch0.3-brightgreen.svg)](https://pytorch.org/)
[![built with ViZDoom](https://img.shields.io/badge/build%20with-ViZDoom1.1.5-blue.svg)](http://vizdoom.cs.put.edu.pl/)
[![built with Sacred 0.7](https://img.shields.io/badge/build%20with-Sacred0.7-yellow.svg)](https://github.com/IDSIA/sacred)

----------------------------------------------

- [x] CRLMaze .wad and .cfg files
- [x] CRLMaze core code-base
- [x] Configurations files to reproduce the experiments
- [x] **Setup scripts and getting started**

In this page we provide the code and all the resources related to the paper **Continual Reinforcement Learning in 3D Non-stationary Environments**. If you plan to use some of the resources you'll find in this page, please **cite our [latest paper](https://arxiv.org/abs/1705.03550)**: 

	@article{lomonaco2019,
       title={Continual Reinforcement Learning in 3D Non-stationary Environments},
       author={Lomonaco, Vincenzo and Desai, Karen and Culurciello, Eugenio and Maltoni, Davide},
       journal={arXiv preprint arXiv:1905.10112},
       year={2019}
	}

----------------------------------------------

## Dependencies

In order to extecute the code in the repository you'll need to install the following dependencies in a [Python 3.x](https://www.python.org/) environment:

* Basic dependences:

```bash
pip3 install numpy scipy matplotlib sacred pymongo
```

* [ViZDoom](https://pypi.python.org/pypi/numpy/1.6.1): _RL API wrapper to ZDoom_

```bash
# Install VizDoom
RUN git clone https://github.com/mwydmuch/ViZDoom ${HOME_DIR}/vizdoom
RUN pip3 install ${HOME_DIR}/vizdoom
```

* [Sacred](https://github.com/IDSIA/sacred): _Experiments Manager_

```bash
pip install sacred
```

* [PyTorch](https://pytorch.org/): _Deep Learning framework_

```bash
pip3 install http://download.pytorch.org/whl/cu80/torch-0.3.1-cp35-cp35m-linux_x86_64.whl
pip3 install torchvision
```

----------------------------------------------

## Project Structure
Up to now the projects is structured as follows:

- [`src/`](src): In this folder you can find all the experiments configurations and the caffe definition files. sI, sII and sIII stand for the NI, NC and NIC scenarios, respectively.
- [`cfgs/`](cfgs): The actual code of the benchmark.
- [`artifacts/`](artifacts): After the setup it will be created and filled with data needed for the experiments. It will also be used for storing partial computations.
- [`scripts/`](script): Scripts for the easy setup and run.
- [`LICENSE`](LICENSE): Standard Creative Commons Attribution 4.0 International License.
- [`README.md`](README.md): This instructions file.
----------------------------------------------

## Getting Started

First of all, let's clone the repository:

```bash
git clone https://github.com/vlomonaco/crlmaze.git
```

Then, in order to run the experiments reported in the paper: 

```bash
cd clrmaze
sudo chmod a+x scripts/*
./scripts/setup.sh
```

After this initial step you can directly run the experiments with the bash scripts [`./scripts/run_exps.sh`](run_exps.sh) for CRL implemented baselines. Since this experiments can take a while (also more than 40h) you can also disable some experiments just by commenting them in the bash script.

----------------------------------------------

## Troubleshooting

- If you find different results from out benchmark (for a few percentage points) _that is to be expected_! First of all because we use the `cudnn` engine which is not fully deterministic for convolutions. Second because the error may be accumulated during the incremental learning process. If you want full reproducibility `"backend"` parameter in the configuration files to `"CPU"`.

- Hey! If you find any trouble don't get frustrated, just ask, we'll answer in a few hours! :-)

----------------------------------------------

## License

This work is licensed under a <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>. 

----------------------------------------------

## Author

* [Vincenzo Lomonaco](http://vincenzolomonaco.com) - email: *vincenzo.lomonaco@unibo.it*
* [Karan Desai](https://kdexd.github.io/) - email: *kdexd@gatech.edu*

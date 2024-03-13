<br>
<p align="center">
<h1 align="center"><strong>RaTrack: Moving Object Detection and Tracking with 4D Radar Point Cloud</strong></h1>
  <p align="center">
    <a href='https://scholar.google.com/citations?user=MbzyV9YAAAAJ&hl=en' target='_blank'>Zhijun Pan¹*</a>&emsp;
    <a href='https://toytiny.github.io/' target='_blank'>Fangqiang Ding²*</a>&emsp;
    <a href='https://www.linkedin.com/in/hantao-zhong/' target='_blank'>Hantao Zhong³*</a>&emsp;
    <a href='https://christopherlu.github.io/' target='_blank'>Chris Xiaoxuan Lu⁴
    </a>&emsp;
    <br>
    *Equal Contribution
    <br>
    Royal College of Art¹&emsp;University of Edinburgh²&emsp;University of Cambridge³&emsp;
    University College London⁴
  </p>
</p>

<div id="top" align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2309.09737-b31b1b.svg)](https://arxiv.org/abs/2309.09737)
[![](https://img.shields.io/youtube/views/IxfyCWyNhfw?label=Demo&style=flat)](https://www.youtube.com/watch?v=IxfyCWyNhfw&feature=youtu.be)
[![GitHub license](https://img.shields.io/badge/license-MIT-green.svg)](https://github.com/LJacksonPan/RaTrack/blob/master/LICENSE)

</div>

<p align="center">
<img src='./doc/ratrack_pipeline.png' width="840">
</p>



## 🔥 News
 - [2024-01-29] Our paper is accepted by [ICRA 2024](https://2024.ieee-icra.org/) 🎉.
 - [2024-01-29] Our paper can be seen here 👉 [arXiv](https://arxiv.org/abs/2309.09737)
 - [2024-03-13] We further improve the overall performance. Please check [Evaluation](#evaluation).
 - [2024-03-13] Our paper demo video can be seen here 👉[video](https://youtu.be/IxfyCWyNhfw)
## 🔗 Citation
If you find our work useful in your research, please consider citing:


```bibtex
@article{pan2023moving,
  title={Moving Object Detection and Tracking with 4D Radar Point Cloud},
  author={Pan, Zhijun and Ding, Fangqiang and Zhong, Hantao and Lu, Chris Xiaoxuan},
  journal={arXiv preprint arXiv:2309.09737},
  year={2023}
}
```

## 📊 Qualitative results
Here are some GIFs to show our qualitative results on tracking. For more qualitative results, please refer to our [demo video](#demo-video)

<p align="center">
<img src='./doc/ratrack_gif1_slow.gif' width="840">
</p>

<p align="center">
<img src='./doc/ratrack_gif2_slow.gif' width="840">
</p>


## ✅ Getting Started

Please ensure you running with an Ubuntu machine with Nvidia GPU (at least 2GB VRAM). 
The code is tested with Ubuntu22.04, and CUDA 11.8 with RTX 4090. Any other machine is not guaranteed to work.

To start, please ensure you have miniconda installed by following the official instructions [here](https://docs.anaconda.com/free/miniconda/miniconda-install/). 

First, clone the repository with the following command and navigate to the root directory of the project:

```bash
git clone git@github.com:LJacksonPan/RaTrack.git
cd RaTrack
```
Create a **RaTrack** environment with the following command:

```bash
conda env create -f environment.yml
```

This will setup a conda environment named `RaTrack` with CUDA 11.8, PyTorch2.2.0.

Installing the pointnet2 pytorch dependencies:

```bash
cd lib
python setup.py install
```

To train the model, please run:

```bash
python main.py
```
This will use the configuration file `config.yaml` to train the model.

To evaluate the model and generate the model predictions, please run:

```bash
python main.py --config configs_eval.yaml
```

## 🔎 Evaluation

To evaluate with the trained RaTrack model, please open the `configs_eval.yaml` and change the `model_path` to the path of the trained model. 
```yaml
model_path: 'checkpoint/track4d_radar/models/model.last.t7'
```


Then run the following command:

```bash
python main.py --config configs_eval.yaml
```

This will generate the predictions in the `results` folder.

If you are interested in evaluating the predictions with our version of AB3DMOT evaluation, please contact us.

## 👏 Acknowledgements

We use the following open-source projects in our work:

- [Pointnet2.Pytorch](https://github.com/sshaoshuai/Pointnet2.PyTorch): We use the pytorch cuda implmentation of pointnet2 module.
- [view-of-delft-dataset](https://github.com/tudelft-iv/view-of-delft-dataset): We the documentation and development kit of the View of Delft (VoD) dataset to develop the model.
- [AB3DMOT](https://github.com/open-mmlab/OpenPCDet): we use AB3DMOT for evaluation metrics.
- [OpenPCDet](https://github.com/open-mmlab/OpenPCDet): we use OpenPCDet for baseline detetion model training and evaluating.
# Repository for Real-to-Sim Tactile Image Transfer and Sim-to-Real Tactile Policy Application
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

**This repository is primarily served for these papers:**
<!-- [Project Website](https://sites.google.com/my.bristol.ac.uk/tactile-gym-sim2real/home) &nbsp;&nbsp;• -->
*Tactile Gym 2.0: Sim-to-real Deep Reinforcement Learning for Comparing Low-cost High-Resolution Robot Touch* 

[Project Website](https://sites.google.com/view/tactile-gym-2/) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://ieeexplore.ieee.org/abstract/document/9847020)

*Optical Tactile Sim-to-Real Policy Transfer via Real-to-Sim Tactile Image Translation*

[Project Website](https://sites.google.com/view/tactile-gym-1/) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](http://arxiv.org/abs/2106.08796)

## In a nut shell

This repo mainly contains code for 
1. Data collection: Collecting pairwise sim and real tactile images (same contact pose);
2. Real-to-Sim Tactile Images Domain Adaptation: Training a pix2pix-GAN for translating real tactile images into sim tactile images;
3. Sim-to-Real Deep-RL Policy Application: Applying deep-RL policis trained in a simulated environment (Tactile Gym) to the real world, using the aforementioned real-to-sim tactile image transfer.

*Real-to-Sim Tactile Images Domain Adaptation (Left: Digitac, Right: DIGIT)*:
<p align="center">
  <img width="300" src="figures/digitac_s2r.gif">
  <img width="300" src="figures/digit_s2r.gif"> <br> 
</p>

*Sim-to-Real Deep-RL Policy Applications (TacTip)*:
<p align="center">
  <img width="768" alt="Sim-to-Real Deep-RL Policies Application" src="figures/tg_1_s2r.gif">
</p>

### Content ###
- [Installation](#installation)
- [Available Tactile Sensors](#available-tactile-sensors)
- [Data Collection](#data-collection)
- [Real-to-Sim Domain Adaptation](#real-to-sim-domain-adaptation)
- [Sim-to-Real Deep-RL Policy Application](#sim-to-real-deep-rl-policy-application)
- [More Interesting Applications](#more-interesting-applications)
- [Contributors](#contributors)
- [Bibtex](#bibtex)



### Installation ###
This repo has only been developed and tested with Ubuntu 18.04 and python 3.8.

```console
git clone https://github.com/yijionglin/tactile_gym_sim2real.git
cd tactile_gym_sim2real
python setup.py install
```

### Available Tactile Sensors ###

This repository works for three tactile sensors:

- `TacTip`: [Tactip](https://www.liebertpub.com/doi/full/10.1089/soro.2017.0052) is a soft, curved, 3D-printed tactile skin with an internal array of pins tipped with markers, which are used to amplify the surface deformation from physical contact against a stimulus.
- `DIGIT`: [DIGIT](https://digit.ml/) shares the same principle of the [Gelsight tactile sensor](https://www.mdpi.com/1424-8220/17/12/2762), but can be fabricated at low cost and is of a size suitable for integration of some robotic hands, such as on the fingertips of the Allegro.
- `DigiTac`: DigiTac is an adapted version of the DIGIT and the TacTip, whereby the 3D-printed skin of a TacTip is customized to integrated onto the DIGIT housing, while keeping the camera and lighting system. In other words, this sensor outputs tactile images of the same dimension as the DIGIT, but with a soft biomimetic skin like other TacTip sensors.


### Data Collection ###

With this repository, you are able to collect pose-labeled tactile images of three types of geometrical features by contacting the tactile sensor with these stimuli: a flat edge, a flat surface, and a set of spherical probes. Note that the transfer model can generalize to unseen curve edges and surfaces, which are explained in our [Paper](https://ieeexplore.ieee.org/abstract/document/9847020).


**a) Real Tactile Images Collection** 

*Note that you should install the [Common Robot Interface](https://github.com/jlloyd237/cri) to run our code for the real robot.*

Navigate to the directory for real tactile images collection,
```
cd tactile_gym_sim2real/data_collection/real
```
Choose one of the contact features you would like to collect for tactile images. For example, for edge feature,
```
cd edge_2d
```
Run the script for data collection for edge,
```
python collect_edge_data_rand.py
```

**b) Simulated Tactile Images Collection** 

*Note that you should install the [Tactile Gym](https://github.com/ac-93/tactile_gym) to run our code for the simulated robot.*

*Note that you should finish collecting the real tactile images before starting to collect simulated ones, as the simulated and real tactile images need to be matched up for later transfer model training.*

Navigate to the directory for sim tactile images collection,
```
cd tactile_gym_sim2real/data_collection/sim
```
Choose the contact features you would like to collect for tactile images. For example, for edge feature,
```
cd edge_2d
```
Run the script for data collection for edge,
```
python quick_collect.py
```

Additionally, you can run the ```test_edge_collect.py``` if you would like to visualize the data collection process in simulation.



### Real-to-Sim Domain Adaptation ###
Once both the sim and the real tactile images are collected, you are able to train a pix2pix-GAN for real-to-sim tactile images translation. For reference, we based this pix2pix-GAN implementation off of the pytorch implementation available [here](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/pix2pix).


Navigate to the directory for real-to-sim domain adaptation,
```
cd tactile_gym_sim2real/pix2pix
```

Before training, it's always good to check the collected pairwise real and simulated tactile images by reviewing them for quality and consistency,

```
python demo_image_generation.py
```

If the dataset looks good, then you can start to train a real-to-sim trasnfer by
```
python pix2pix.py
```



### Sim-to-Real Deep-RL Policy Application ###

*Note that before the sim-to-real application, you should already have an image transfer model for real-to-sim mapping obtained from previous steps and a deep-RL policy trained for a specific task using [Tactile Gym](https://github.com/ac-93/tactile_gym).*

Navigate to the directory for sim-to-real deep-RL policy application,
```
cd online_experiments
```

Before any sim-to-real application, it's always good to check the translation quality of the transfer model by
```
python test_gan.py
```

Choose the task you would like to play with; for example, if you want the robot to perform the edge-following task,


```
cd edge_follow_env
python evaluate_edge_follow.py
```


### Contributors ###

[Alex Church](https://scholar.google.com/citations?user=D7eIiqwAAAAJ&hl=en&oi=ao)

[Yijiong Lin](https://yijionglin.github.io/) (<ins>Currently looking for a PostDoc or Fellowship position starting around 2025 January</ins>)


### More Interesting Applications ###
These papers are (partly) built on top of this repository:

*1. Bi-Touch: Bimanual Tactile Manipulation with Sim-to-Real Deep Reinforcement Learning*

[Project Website](https://sites.google.com/view/bi-touch/) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://ieeexplore.ieee.org/abstract/document/10184426)

*2. Attention for Robot Touch: Tactile Saliency Prediction for Robust Sim-to-Real Tactile Control* 

[Project Website](https://sites.google.com/view/tactile-saliency/) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://arxiv.org/pdf/2307.14510.pdf)

*3. Sim-to-Real Model-Based and Model-Free Deep Reinforcement Learning for Tactile Pushing*

[Project Website](https://sites.google.com/view/tactile-rl-pushing/) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://sites.google.com/view/tactile-rl-pushing/)

*4. TouchSDF: A DeepSDF Approach for 3D Shape Reconstruction Using Vision-Based Tactile Sensing*

[Project Website](https://touchsdf.github.io/) &nbsp;&nbsp;•&nbsp;&nbsp;[Paper](https://touchsdf.github.io/)

### Bibtex ###

If you use this repo in your work, please cite

```
@ARTICLE{lin2022tactilegym2,
  author={Lin, Yijiong and Lloyd, John and Church, Alex and Lepora, Nathan F.},
  journal={IEEE Robotics and Automation Letters}, 
  title={Tactile Gym 2.0: Sim-to-Real Deep Reinforcement Learning for Comparing Low-Cost High-Resolution Robot Touch}, 
  year={2022},
  volume={7},
  number={4},
  pages={10754-10761},
  doi={10.1109/LRA.2022.3195195},
  url={https://ieeexplore.ieee.org/abstract/document/9847020},
  }

@InProceedings{church2021optical,
     title={Tactile Sim-to-Real Policy Transfer via Real-to-Sim Image Translation},
     author={Church, Alex and Lloyd, John and Hadsell, Raia and Lepora, Nathan F.},
     booktitle={Proceedings of the 5th Conference on Robot Learning}, 
     year={2022},
     editor={Faust, Aleksandra and Hsu, David and Neumann, Gerhard},
     volume={164},
     series={Proceedings of Machine Learning Research},
     month={08--11 Nov},
     publisher={PMLR},
     pdf={https://proceedings.mlr.press/v164/church22a/church22a.pdf},
     url={https://proceedings.mlr.press/v164/church22a.html},
}
```

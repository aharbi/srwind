# Super-resolution of NREL Wind Integration National Dataset (WIND) Toolkit

Super-resolution of wind data using machine learning techniques. The repository contains an implementation of two methods:
- Local Regression: Implementation of the algorithm in ["Fast and Accurate Image Upscaling with Super-Resolution Forests"](https://openaccess.thecvf.com/content_cvpr_2015/papers/Schulter_Fast_and_Accurate_2015_CVPR_paper.pdf) using a ridge regression and a random forest model.
- Denoising Diffusion Models: Implementation of the algorithm in ["Denoising Diffusion Probabilistic Models"](https://arxiv.org/pdf/2006.11239.pdf) with modification to handle super-resolution from ["Image Super-Resolution via Iterative Refinement"](https://arxiv.org/pdf/2104.07636.pdf). Trained model weights are available in the repository.

## Acknowledgements
The functions unet.py and sr3.py are adapted from the implementation at: https://github.com/TeaPearce/Conditional_Diffusion_MNIST.
The functions energy_spectrum.py, kinetic_energy_initial.py, and kinetic_energy_initial_with_post_normalization.py are adapted from the implementation at: https://github.com/RupaKurinchiVendhan/WiSoSuper/blob/main/energy.py.

The SR3 algorithm was originally put forth by Saharia et al.: 
[1] C. Saharia, J. Ho, W. Chan, T. Salimans, D. J. Fleet, and M. Norouzi, “Image Super-Resolution Via Iterative Refinement,” IEEE Trans. Pattern Anal. Mach. Intell., pp. 1–14, 2022, doi: 10.1109/TPAMI.2022.3204461.



## Operating the Repository
First, construct a python environment and set up the directory using the "environment.yml" file. If the environment.yml file doesn't work, you need at least the following packages from various sources:

* numpy
* scipy
* matplotlib
* scikit-learn
* scikit-image
* pytorch
* gdown
* tqdm
* p7zip

Next, run the bash script ''dataset.sh.'' This will download the NREL dataset as used by the WiSoSuper Resolution paper. NOTE: there is a lot of data here. If only intending to run testing and validation, the training dataset can be commented out and is unnecessary.

### Descriptions of Functions
* <b>exp.py</b>: functions to carry out main experiments in the code.
* <b>metrics.py</b>: stores the implementations of most metrics.
* <b>energy_spectrum.py</b>: implementation of the kinetic energy spectrum analysis, used to create Fig. 3 in our report.
* <b>cosine_similarity.py</b>: implementation of the cosine similarity metrics and analysis, used to create Fig. 4 in our report.

## Sample Outputs
![title](figures/1.png)
![title](figures/2.png)
![title](figures/3.png)
![title](figures/4.png)
![title](figures/5.png)
![title](figures/6.png)
![title](figures/7.png)
![title](figures/8.png)
![title](figures/9.png)
![title](figures/10.png)
![title](figures/11.png)
![title](figures/12.png)
![title](figures/13.png)
![title](figures/14.png)
![title](figures/15.png)
![title](figures/16.png)
![title](figures/17.png)
![title](figures/18.png)
![title](figures/19.png)
![title](figures/20.png)

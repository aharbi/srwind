import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import scipy.integrate as ig
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
import os
import random
from PIL import Image
import argparse
import dataset
import visualization 
import json

"""
This code is adpated from the implementation in:
https://github.com/RupaKurinchiVendhan/WiSoSuper/blob/main/energy.py
"""

Energy_Spectrum = {'HR':  {'x':[], 'y':[]}, 
                       'LR':  {'x':[], 'y':[]}, 
                        'Bicubic':  {'x':[], 'y':[]}, 
                        'Ridge Regression':  {'x':[], 'y':[]}, 
                        'Random Forest':  {'x':[], 'y':[]}, 
                        'SR3 (Regression)':  {'x':[], 'y':[]}, 
                        'SR3 (Diffusion)':  {'x':[], 'y':[]}
                       }

def kinetic_energy_spectra(
        current_data_matrix, 
        current_label_matrix, 
        prediction_bi, 
        prediction_rr, 
        prediction_rf, 
        prediction_reg_sr3,
        prediction_diff_sr3
    ):
    
    def energy_spectrum(img):

        img.save('greyscale.png')
        image = mpimg.imread("greyscale.png")

        npix = image.shape[0]
        fourier_image = np.fft.fftn(image)
        fourier_amplitudes = np.abs(fourier_image)**2
        fourier_amplitudes = np.fft.fftshift(fourier_amplitudes)

        kfreq = np.fft.fftfreq(npix) * npix
        kfreq2D = np.meshgrid(kfreq, kfreq)
        knrm = np.sqrt(kfreq2D[0]**2 + kfreq2D[1]**2)

        knrm = knrm.flatten()
        fourier_amplitudes = fourier_amplitudes.flatten()

        kbins = np.arange(0.5, npix//2+1, 1.)
        kvals = (kbins[1:] + kbins[:-1])
        Abins, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                            statistic = "mean",
                                            bins = kbins)
        Abins *= np.pi * (kbins[1:]**2 - kbins[:-1]**2)

        return kvals, Abins


    def compare_outputs():
        for i in range(2):
            min = current_label_matrix[i,:,:].min()
            max = current_label_matrix[i,:,:].max()
            
            wind_profile = current_label_matrix[i,:,:]
            # Normalization against itself (to capture variations)
            wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['HR']['x'].append(HR_kvals2)
            Energy_Spectrum['HR']['y'].append(HR_ek)
    
            wind_profile = current_data_matrix[i,:,:]
            # Normalization against itself (to capture variations)        
            wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['LR']['x'].append(HR_kvals2)
            Energy_Spectrum['LR']['y'].append(HR_ek)
    
            wind_profile = prediction_bi[i,:,:]
            # Normalization against itself (to capture variations)
            wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['Bicubic']['x'].append(HR_kvals2)
            Energy_Spectrum['Bicubic']['y'].append(HR_ek)
    
            wind_profile = prediction_rr[i,:,:]
            # Normalization against itself (to capture variations)
            # wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['Ridge Regression']['x'].append(HR_kvals2)
            Energy_Spectrum['Ridge Regression']['y'].append(HR_ek)
    
            wind_profile = prediction_rf[i,:,:]
            # Normalization against itself (to capture variations)
            wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['Random Forest']['x'].append(HR_kvals2)
            Energy_Spectrum['Random Forest']['y'].append(HR_ek)
    
            wind_profile = prediction_reg_sr3[i,:,:]
            # Normalization against itself (to capture variations)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['SR3 (Regression)']['x'].append(HR_kvals2)
            Energy_Spectrum['SR3 (Regression)']['y'].append(HR_ek)
    
            wind_profile = prediction_diff_sr3[i,:,:]
            # Normalization against itself (to capture variations)
            # wind_profile_normalized = (wind_profile * 255).astype(np.uint8)
            wind_profile_normalized = ((wind_profile - wind_profile.min()) / (wind_profile.max() - wind_profile.min()) * 255).astype(np.uint8)
            # Alternative: Normalization against the HR image
            # wind_profile_normalized = ((wind_profile - min) / (max - min) * 255).astype(np.uint8)
            image = Image.fromarray(wind_profile_normalized)
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['SR3 (Diffusion)']['x'].append(HR_kvals2)
            Energy_Spectrum['SR3 (Diffusion)']['y'].append(HR_ek)

    compare_outputs()


def plot_energy_spectra(fname="wind_spectrum_norm"):
    colors = {'HR': 'black', 'LR': 'pink','Bicubic': 'tab:blue', 'Ridge Regression': 'tab:orange', 'Random Forest': 'tab:green', 'SR3 (Regression)': 'tab:red', 'SR3 (Diffusion)': 'tab:purple'}
    
    for model in Energy_Spectrum:
        k = np.flip(np.mean(Energy_Spectrum[model]['x'], axis=0))
        E = np.mean(Energy_Spectrum[model]['y'], axis=0) / 10000

        # add normalization
        totalEnergy = ig.trapezoid(np.flip(E), np.flip(k))
        print("\nTotal Energies: ")
        print(totalEnergy)
        plt.loglog(k, E/totalEnergy, color=colors[model], label=model)

    plt.xlabel("k (wavenumber)")
    plt.ylabel("Kinetic Energy")
    plt.tight_layout()
    plt.title("Energy Spectrum")
    plt.legend()
    plt.savefig(fname+".png", dpi=1000, transparent=True, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numImgs", type=int, default=100)
    parser.add_argument("--fname", default="wind_spectrum_norm")
    args = parser.parse_args()

    num_images = args.numImgs

    for i in range(num_images):
        print("\nImage: {}/{}".format(i+1,num_images))
        current_data_matrix, current_label_matrix, prediction_bi, prediction_rr, prediction_rf, prediction_reg_sr3, prediction_diff_sr3 = visualization.compute_random_result()

        
        kinetic_energy_spectra(
            current_data_matrix[0,:,:,:], 
            current_label_matrix[0,:,:,:], 
            prediction_bi[0,:,:,:], 
            prediction_rr[0,:,:,:], 
            prediction_rf[0,:,:,:], 
            prediction_reg_sr3[0,:,:,:],
            prediction_diff_sr3[0,:,:,:]
        )
        
    saveName = args.fname+"_{}".format(num_images)
    plot_energy_spectra(fname=saveName)

    # save norm data
    # saveDR = "./physics_metrics_server/"
    # saveName = args.fname+"_{}".format(num_images)
    # f = open(saveDR+saveName+".json", "w")
    # json.dump(Energy_Spectrum, f, sort_keys=True, indent=2)
    # f.close()



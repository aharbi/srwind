import matplotlib.image as mpimg
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
import os
import random
from PIL import Image

import dataset

"""
This code is adpated from the implementation in:
https://github.com/RupaKurinchiVendhan/WiSoSuper/blob/main/energy.py
"""

def kinetic_energy_spectra(
        current_data_matrix, 
        current_label_matrix, 
        # prediction_bi, 
        # prediction_rr, 
        # prediction_rf, 
        # prediction_reg_sr3,
        # prediction_diff_sr3
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

    def plot_energy_spectra():
        colors = {'HR': 'black', 'LR': 'pink', 
                  #'Bicubic': 'tab:blue', 'Ridge Regression': 'tab:orange', 'Random Forest': 'tab:green', 'SR3 (Regression)': 'tab:red', 'SR3 (Diffusion)': 'tab:purple'
                  }
        for model in Energy_Spectrum:
            k = np.flip(np.mean(Energy_Spectrum[model]['x'], axis=0))
            E = np.mean(Energy_Spectrum[model]['y'], axis=0) / 10000
            plt.loglog(k, E, color=colors[model], label=model)
        plt.xlabel("k (wavenumber)")
        plt.ylabel("Kinetic Energy")
        plt.tight_layout()
        plt.title("Energy Spectrum")
        plt.legend()
        plt.savefig("wind_spectrum.png", dpi=1000, transparent=True, bbox_inches='tight')
        plt.show()

    Energy_Spectrum = {'HR':  {'x':[], 'y':[]}, 
                       'LR':  {'x':[], 'y':[]}, 
                    #    'Bicubic':  {'x':[], 'y':[]}, 
                    #    'Ridge Regression':  {'x':[], 'y':[]}, 
                    #    'Random Forest':  {'x':[], 'y':[]}, 
                    #    'SR3 (Regression)':  {'x':[], 'y':[]}, 
                    #    'SR3 (Diffusion)':  {'x':[], 'y':[]}
                       }
    for i in range(2):
        wind_profile = current_label_matrix[0,i,:,:]
        HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        Energy_Spectrum['HR']['x'].append(HR_kvals2)
        Energy_Spectrum['HR']['y'].append(HR_ek)

        wind_profile = current_data_matrix[0,i,:,:]
        HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        Energy_Spectrum['LR']['x'].append(HR_kvals2)
        Energy_Spectrum['LR']['y'].append(HR_ek)

        # wind_profile = prediction_bi[i,:,:]
        # HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        # Energy_Spectrum['Bicubic']['x'].append(HR_kvals2)
        # Energy_Spectrum['Bicubic']['y'].append(HR_ek)

        # wind_profile = prediction_rr[i,:,:]
        # HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        # Energy_Spectrum['Ridge Regression']['x'].append(HR_kvals2)
        # Energy_Spectrum['Ridge Regression']['y'].append(HR_ek)

        # wind_profile = prediction_rf[i,:,:]
        # HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        # Energy_Spectrum['Random Forest']['x'].append(HR_kvals2)
        # Energy_Spectrum['Random Forest']['y'].append(HR_ek)

        # wind_profile = prediction_diff_sr3[i,:,:]
        # HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        # Energy_Spectrum['SR3 (Regression)']['x'].append(HR_kvals2)
        # Energy_Spectrum['SR3 (Regression)']['y'].append(HR_ek)

        # wind_profile = prediction_reg_sr3[i,:,:]
        # HR_kvals2, HR_ek = energy_spectrum(Image.fromarray(wind_profile.astype('uint8')))
        # Energy_Spectrum['SR3 (Diffusion)']['x'].append(HR_kvals2)
        # Energy_Spectrum['SR3 (Diffusion)']['y'].append(HR_ek)

    plot_energy_spectra()


if __name__ == "__main__":
    path = "dataset/test/"

    file_names = os.listdir(path)
    file_name = random.choice(file_names)
    i = random.choice(range(256))

    current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
        os.path.join(path, file_name)
    )

    current_data_matrix = current_data_matrix[i : i + 1]
    current_label_matrix = current_label_matrix[i : i + 1]

    kinetic_energy_spectra(
            current_data_matrix, 
            current_label_matrix,
        )

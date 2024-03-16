
import numpy as np
import scipy as sp
import scipy.fft as spf
import scipy.stats as stats
import scipy.integrate as ig
import matplotlib.pyplot as plt
import json
import argparse
import visualization
import pdb

"""
This code is a simplified form of the implementation in:
https://github.com/RupaKurinchiVendhan/WiSoSuper/blob/main/energy.py

It takes in an image, normalizes it to itself, and then computes the FFT. Note that it computes 
the FFT of each row, and then gets the total "k vector" in cycles/pixel frequency space, to that pixel.
At the end, it computes the average frequency content of a bin of values. These are "cycles/pixel" values, 
which we compute as the inverse of pixel space. These are NOT k-vectors, as the original implementation
suggests. We also then normalize to "total energy" to match the scales of all models, which is again 
absent from the original implementation. We ignore some of the more complicated lines of code in the
original implementation, which seem unreasonable to include.

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
        prediction_diff_sr3, 
        fname="./wind_spectrum_norm"
    ):
    
    def energy_spectrum(HR_patch: np.ndarray):
        """Computes PSD of image. Takes slices of images for comparison and then 
        averages overa all slices.

        (alternative approach: vectorize entire patch and take single fft.)

        Args:

        Returns:
            ndarray: 1xnd vector containing the averaged ref PSD content
            ndarray: 1xnd vector containing the averaged hr PSD content
            ndarray: 1xnd vector containing the "frequency" values
        """

        # get frequency axis
        xdims = HR_patch.shape[1] # returns pixels along x-axis
        ydims = HR_patch.shape[0] # gets number of rows in patch
        freqs = spf.fftfreq(n=xdims)  # generate frequency domain sample points [cycles/pixel]

        # normalize each patch to self
        hr_min = np.min(HR_patch)
        hr_max = np.max(HR_patch)
        
        # pdb.set_trace()
        # HR_patch = np.divide(HR_patch-hr_min, hr_max-hr_min)
        # pdb.set_trace()

        # get power spectrum
        hr_psd = np.square(np.abs(spf.fftn(HR_patch)))

        # pdb.set_trace()
        # putting kVals into 2D space
        freqsX, freqsY = np.meshgrid(freqs,freqs) # returns "grid" of k0k0, k0k1, k0k2,...; k1k0, k1k1, ... etc.
        kvals = np.sqrt(freqsX**2 + freqsY**2) # get vector of "distances" to each pixel - from WiSo paper

        # flatten everythign into vectors
        kvals = kvals.flatten()
        hr_psd = hr_psd.flatten()

        numBins = 51
        kbins = np.linspace(0, kvals.max(), numBins)

        hist_hr,_,_ = stats.binned_statistic(kvals, hr_psd, statistic="mean", bins=kbins)
        
        # pdb.set_trace()
        # return everything
        return kbins, hist_hr


    def compare_outputs():
        for i in range(2):
            min = current_label_matrix[i,:,:].min()
            max = current_label_matrix[i,:,:].max()
            
            # HIGH RES
            image = current_label_matrix[i,:,:]            
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['HR']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['HR']['y'].append(HR_ek)
    
            # LOW RES
            image = current_data_matrix[i,:,:]
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['LR']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['LR']['y'].append(HR_ek)
    
            # BICUBIC
            image = prediction_bi[i,:,:]            
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['Bicubic']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['Bicubic']['y'].append(HR_ek)
    
            image = prediction_rr[i,:,:]            
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['Ridge Regression']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['Ridge Regression']['y'].append(HR_ek)
    
            image = prediction_rf[i,:,:]            
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['Random Forest']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['Random Forest']['y'].append(HR_ek)
    
            image = prediction_reg_sr3[i,:,:]            
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['SR3 (Regression)']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['SR3 (Regression)']['y'].append(HR_ek)
    
            image = prediction_diff_sr3[i,:,:]        
            HR_kvals2, HR_ek = energy_spectrum(image)
            Energy_Spectrum['SR3 (Diffusion)']['x'].append(HR_kvals2.tolist())
            Energy_Spectrum['SR3 (Diffusion)']['y'].append(HR_ek)


    compare_outputs()        

    return


def plot_energy_spectra(fname="./wind_spectrum_norm"):
    colors = {'HR': 'black', 'LR': 'pink','Bicubic': 'tab:blue', 'Ridge Regression': 'tab:orange', 'Random Forest': 'tab:green', 'SR3 (Regression)': 'tab:red', 'SR3 (Diffusion)': 'tab:purple'}
    for i in range(2):
        # f = open(fname+"_ch{}.json".format(i), "r")
        # Energy_Spectrum = json.load(f)
        # f.close()

        for model in Energy_Spectrum:
            k = (np.mean(Energy_Spectrum[model]['x'], axis=0))
            E = np.mean(Energy_Spectrum[model]['y'], axis=0)

            # add normalization
            pdb.set_trace()
            totalEnergy = ig.trapezoid(E, k[:-1])
            print("\nTotal Energies: ")
            print(totalEnergy)
            plt.loglog(k, E/totalEnergy, color=colors[model], label=model)

        plt.xlabel("k (cycles/pixel)")
        plt.ylabel("Kinetic Energy")
        plt.tight_layout()
        plt.title("Energy Spectrum")
        plt.legend()
        plt.savefig(fname+"_ch{}.png".format(i), dpi=1000, transparent=False, bbox_inches='tight')
        # plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--numImgs", type=int, default=100)
    parser.add_argument("--savePath", default="wind_spectrum")
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
            prediction_diff_sr3[0,:,:,:],
        )
        
    fname=args.savePath
    pdb.set_trace()
    g = open(fname+"_ch{}.json".format(i), "w")
    json.dump(Energy_Spectrum, g, sort_keys=True, indent=2)   
    g.close() 

    plot_energy_spectra(fname=args.savePath)
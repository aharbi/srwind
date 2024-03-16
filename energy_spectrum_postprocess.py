
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

def main(args):
    pathToData = "./final data/KEv2_selfNorm_100_31bins.json"
    f = open(pathToData, "r")
    Energy_Spectrum = json.load(f)
    f.close()

    plot_energy_spectra(fname=args.savePath, Energy_Spectrum=Energy_Spectrum)
    

def plot_energy_spectra(fname="./wind_spectrum_norm", Energy_Spectrum={}):
    colors = {'HR': 'black', 'LR': 'pink','Bicubic': 'tab:blue', 'Ridge Regression': 'tab:orange', 'Random Forest': 'tab:green', 'SR3 (Regression)': 'tab:red', 'SR3 (Diffusion)': 'tab:purple'}

    for model in Energy_Spectrum:
        if not(model=="LR"):
            k = (np.mean(Energy_Spectrum[model]['x'], axis=0))
            E = np.mean(Energy_Spectrum[model]['y'], axis=0)

            # add normalization
            # pdb.set_trace()
            ksub = k[:-1]
            totalEnergy = ig.trapezoid(E, ksub)
            print("\nTotal Energies: ")
            print(totalEnergy)
            plt.loglog(ksub, E/totalEnergy, color=colors[model], label=model)

    plt.xlabel("k (cycles/pixel)")
    plt.ylabel("Energy")
    plt.tight_layout()
    plt.title("Energy Spectrum")
    plt.axis([0, 1, 10e-8, 1])
    plt.legend(loc="lower left")
    plt.savefig(fname+".png", dpi=1000, transparent=False, bbox_inches='tight')
    plt.show()

    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--savePath", default="wind_spectrum")
    args = parser.parse_args()

    main(args)
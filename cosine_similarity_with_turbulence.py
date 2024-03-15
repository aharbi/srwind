import numpy as np
import util
import dataset
import llr
import sr3
import metrics
import os
import matplotlib.pyplot as plt
import scipy.ndimage as nd
import scipy.fft as spf
import scipy.stats as stats
import pdb;
import visualization
import json
import argparse
from PIL import Image
import matplotlib.image as mpimg

"""
This script is similar to the cosine_turbulence.py script. However, this one then subtracts out the averaged matrix values and computes
kinetic energy spectra on the REMAINING parts of the images.

The Kinetic Energy analysis ("energy_spectrum" function) is adapted from code in the repository: https://github.com/RupaKurinchiVendhan/WiSoSuper/blob/main/energy.py.

"""

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

def cos_similarity(ref_patch: np.ndarray, HR_patch: np.ndarray, avgKernel = 1):
    """Computes the cosine similarity error between reference patch and predicted HR_patch,
      after averaging over the specified kernel. This is similar to computing the cosine similarity between
      avg-pooled images, but where dimension is not reduced.
      
    Args:
        ref_patch (np.ndarray): reference high-res patch (2x100x100)
        HR_patch (np.ndarray): predicted high-res patch (2x100x100)
    
    Returns:
        float (1x2): value of cosine similarity.
        ndarray: filtered (averaged) reference patch
        ndarray: filtered (averaged) high-res patch
    """

    def filterAvg(subArray):
        # input: array of dimension 1 x n*d, where n and d are the subarray dimensions passed into scipy filter.
        # output: average over all elements in subArray
                
        aVec = np.ndarray.flatten(subArray)
        return np.mean(aVec)

    # set up result arrays
    dimsPatch = ref_patch.shape
    numChs = dimsPatch[0]

    ref_filt_all = hr_filt_all = np.zeros(dimsPatch)
    overlaps = np.zeros((1,numChs))

    # loop over channels
    for ii in np.arange(numChs):
        # step 1: get scipy generic filter
        # pdb.set_trace()
        ref_filt = nd.generic_filter(input=ref_patch[ii,:,:], function=filterAvg, size=avgKernel, mode="wrap")
        hr_filt = nd.generic_filter(input=HR_patch[ii,:,:], function=filterAvg, size=avgKernel, mode="wrap")

        # step 2: get cos similarity   
        flatRef = ref_filt.flatten()
        flatHR = hr_filt.flatten()
        num = flatRef.dot(flatHR)
        den = np.linalg.norm(flatRef)*np.linalg.norm(flatHR)

        ref_filt_all[ii,:,:] = ref_filt
        hr_filt_all[ii,:,:] = hr_filt
        overlaps[0,ii] = num/den

        # import pdb; pdb.set_trace()

    return overlaps, ref_filt_all, hr_filt_all

def exp_cos_sim(kernel=1, numImgs=10, fname="cos_sim"):
    """
    Experiment Physics Metrics: Performs cosine similarity for different averaging amounts
    and computes the energy spectra for the resulting turbulence values.    
    
    """

    # set up saving and results storage
    saveDR = "./physics_metrics_server/"
    try:
        os.mkdir(saveDR)        
    except:
        print("Save directory already exists.")

    # results storage arrays - each entry is an inference type, with "x" data and "y" data elements (1 average result over batch per kernel), and the std deviations
    Cos_Sim_Dict = {        
        "Bicubic": {"x": [], "y": []},
        "RidgeRegression": {"x": [], "y": []},
        "RandomForest": {"x": [], "y": []},
        "SR3_Reg": {"x": [], "y": []},
        "SR3_Diff": {"x": [], "y": []},
        "kernels": []
    }

    Kin_En_Dict = {        
        "HR": {"k": [], "E": []},
        "Bicubic": {"k": [], "E": []},
        "RidgeRegression": {"k": [], "E": []},
        "RandomForest": {"k": [], "E": []},
        "SR3_Reg": {"k": [], "E": []},
        "SR3_Diff": {"k": [], "E": []},
        "kernels": []
    }

    # pdb.set_trace()

    for ii in np.arange(numImgs):
        # step one: load random test result
        (curr_data, 
        curr_label,
        pred_bi,
        pred_rr,
        pred_rf,
        pred_reg_sr3,
        pred_dif_sr3) = visualization.compute_random_result()
        # pdb.set_trace()
        # step two: iterate cosine_similarity over kernels

        numKernels = 1
        
        # initialize empty arrays to store kernel avg results
        bi_results_x = []
        bi_results_y = []
        rr_results_x = []
        rr_results_y = []
        rf_results_x = []
        rf_results_y = []
        sr3_reg_results_x = []
        sr3_reg_results_y = []
        sr3_dif_results_x = []
        sr3_dif_results_y = []
        

        print("\nImage {} : {}".format(ii+1,numImgs))

        print("    kernel: {}".format(kernel))
        
        # BICUBIC
        overlaps,_,hr_filt = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_bi[0,:,:,:], avgKernel=kernel)
        bi_results_x.append(overlaps[0,0])
        bi_results_y.append(overlaps[0,1])

        # subtract avg.
        turb_x = pred_bi[0,0,:,:]-hr_filt[0,:,:]
        turb_y = pred_bi[0,1,:,:]-hr_filt[1,:,:]

        # normalize
        turb_x_norm = ((turb_x - turb_x.min()) / (turb_x.max() - turb_x.min()) * 255).astype(np.uint8)
        # Alternative: Normalization against the HR image
        # wind_profile_normalized = ((wind_profile_normalized - min) / (max - min) * 255).astype(np.uint8)
        image = Image.fromarray(turb_x_norm)
        HR_kvals2, HR_ek = energy_spectrum(image)
        Kin_En_Dict["Bicubic"]["k"].append(HR_kvals2.tolist())
        Kin_En_Dict["Bicubic"]["E"].append(HR_ek.tolist())    

        # HIGH RESOLUTION IMG
        # subtract avg.
        turb_x = curr_label[0,0,:,:]-hr_filt[0,:,:]
        turb_y = curr_label[0,1,:,:]-hr_filt[1,:,:]

        # normalize
        turb_x_norm = ((turb_x - turb_x.min()) / (turb_x.max() - turb_x.min()) * 255).astype(np.uint8)
        # Alternative: Normalization against the HR image
        # wind_profile_normalized = ((wind_profile_normalized - min) / (max - min) * 255).astype(np.uint8)
        image = Image.fromarray(turb_x_norm)
        HR_kvals2, HR_ek = energy_spectrum(image)
        Kin_En_Dict['HR']["k"].append(HR_kvals2.tolist())
        Kin_En_Dict['HR']["E"].append(HR_ek.tolist())   

        # RIDGE REGRESSION
        overlaps,_,hr_filt = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_rr[0,:,:,:], avgKernel=kernel)
        rr_results_x.append(overlaps[0,0])
        rr_results_y.append(overlaps[0,1])

        # subtract avg.
        turb_x = pred_rr[0,0,:,:]-hr_filt[0,:,:]
        turb_y = pred_rr[0,1,:,:]-hr_filt[1,:,:]

        # normalize
        turb_x_norm = ((turb_x - turb_x.min()) / (turb_x.max() - turb_x.min()) * 255).astype(np.uint8)
        # Alternative: Normalization against the HR image
        # wind_profile_normalized = ((wind_profile_normalized - min) / (max - min) * 255).astype(np.uint8)
        image = Image.fromarray(turb_x_norm)
        HR_kvals2, HR_ek = energy_spectrum(image)
        Kin_En_Dict["RidgeRegression"]['k'].append(HR_kvals2.tolist())
        Kin_En_Dict["RidgeRegression"]['E'].append(HR_ek.tolist())   

        # RANDOM FOREST
        overlaps,_,hr_filt = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_rf[0,:,:,:], avgKernel=kernel)
        rf_results_x.append(overlaps[0,0])
        rf_results_y.append(overlaps[0,1])

        # subtract avg.
        turb_x = pred_rf[0,0,:,:]-hr_filt[0,:,:]
        turb_y = pred_rf[0,1,:,:]-hr_filt[1,:,:]

        # normalize
        turb_x_norm = ((turb_x - turb_x.min()) / (turb_x.max() - turb_x.min()) * 255).astype(np.uint8)
        # Alternative: Normalization against the HR image
        # wind_profile_normalized = ((wind_profile_normalized - min) / (max - min) * 255).astype(np.uint8)
        image = Image.fromarray(turb_x_norm)
        HR_kvals2, HR_ek = energy_spectrum(image)
        Kin_En_Dict["RandomForest"]['k'].append(HR_kvals2.tolist())
        Kin_En_Dict["RandomForest"]['E'].append(HR_ek.tolist())   

        # SR3 REG
        overlaps,_,hr_filt = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_reg_sr3[0,:,:,:], avgKernel=kernel)
        sr3_reg_results_x.append(overlaps[0,0])
        sr3_reg_results_y.append(overlaps[0,1])

        # subtract avg.
        turb_x = pred_reg_sr3[0,0,:,:]-hr_filt[0,:,:]
        turb_y = pred_reg_sr3[0,1,:,:]-hr_filt[1,:,:]

        # normalize
        turb_x_norm = ((turb_x - turb_x.min()) / (turb_x.max() - turb_x.min()) * 255).astype(np.uint8)
        # Alternative: Normalization against the HR image
        # wind_profile_normalized = ((wind_profile_normalized - min) / (max - min) * 255).astype(np.uint8)
        image = Image.fromarray(turb_x_norm)
        HR_kvals2, HR_ek = energy_spectrum(image)
        Kin_En_Dict["SR3_Reg"]['k'].append(HR_kvals2.tolist())
        Kin_En_Dict["SR3_Reg"]['E'].append(HR_ek.tolist())   

        # SR3 DIFF.
        overlaps,_,hr_filt = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_dif_sr3[0,:,:,:], avgKernel=kernel)
        sr3_dif_results_x.append(overlaps[0,0])
        sr3_dif_results_y.append(overlaps[0,1])

        # subtract avg.
        turb_x = pred_dif_sr3[0,0,:,:]-hr_filt[0,:,:]
        turb_y = pred_dif_sr3[0,1,:,:]-hr_filt[1,:,:]

        # normalize
        turb_x_norm = ((turb_x - turb_x.min()) / (turb_x.max() - turb_x.min()) * 255).astype(np.uint8)
        # Alternative: Normalization against the HR image
        # wind_profile_normalized = ((wind_profile_normalized - min) / (max - min) * 255).astype(np.uint8)
        image = Image.fromarray(turb_x_norm)
        HR_kvals2, HR_ek = energy_spectrum(image)
        Kin_En_Dict["SR3_Diff"]['k'].append(HR_kvals2.tolist())
        Kin_En_Dict["SR3_Diff"]['E'].append(HR_ek.tolist())   

        # step three: append results to respective locations in dictionary
        Cos_Sim_Dict["Bicubic"]["x"].append(bi_results_x)
        Cos_Sim_Dict["Bicubic"]["y"].append(bi_results_y)
        Cos_Sim_Dict["RidgeRegression"]["x"].append(rr_results_x)
        Cos_Sim_Dict["RidgeRegression"]["y"].append(rr_results_y)
        Cos_Sim_Dict["RandomForest"]["x"].append(rf_results_x)
        Cos_Sim_Dict["RandomForest"]["y"].append(rf_results_y)
        Cos_Sim_Dict["SR3_Reg"]["x"].append(sr3_reg_results_x)
        Cos_Sim_Dict["SR3_Reg"]["y"].append(sr3_reg_results_y)
        Cos_Sim_Dict["SR3_Diff"]["x"].append(sr3_dif_results_x)
        Cos_Sim_Dict["SR3_Diff"]["y"].append(sr3_dif_results_y)

    Cos_Sim_Dict["kernels"] = kernel

    # step four: write dict to json file in saveDR
    f = open(saveDR+fname+".json", "w")
    json.dump(Cos_Sim_Dict, f, sort_keys=True, indent=2)
    g = open(saveDR+fname+"_kinEn.json", "w")
    json.dump(Kin_En_Dict, f, sort_keys=True, indent=2)
    # pdb.set_trace()
    f.close()
    g.close()

    return


def process_cos_sim(saveName="cos_sim_data", loadname="cos_sim_data"):
    # step 1: load the stored json data
    saveDR = "./physics_metrics_server/"
    fname = loadname

    f = open(saveDR+fname, "r")
    Cos_Sim_Dict = json.load(f)
    f.close()

    # pdb.set_trace()

    # step 2: compute averages and std deviations
    avgs = { # NOTE: NAMES OF THE FIELDS IN AVGS MUST MATCH COS_SIM_DICT
        "Bicubic": {"avgs_x": [], "stds_x": [], "avgs_y": [], "stds_y": []},
        "RidgeRegression": {"avgs_x": [], "stds_x": [], "avgs_y": [], "stds_y": []},
        "RandomForest": {"avgs_x": [], "stds_x": [], "avgs_y": [], "stds_y": []},
        "SR3_Reg": {"avgs_x": [], "stds_x": [], "avgs_y": [], "stds_y": []},
        "SR3_Diff": {"avgs_x": [], "stds_x": [], "avgs_y": [], "stds_y": []},
    }
    
    for model in avgs:
        avgs[model]["avgs_x"] = np.mean(Cos_Sim_Dict[model]["x"], axis=0).tolist()
        avgs[model]["stds_x"] = np.std(Cos_Sim_Dict[model]["x"], axis=0).tolist()
        avgs[model]["avgs_y"] = np.mean(Cos_Sim_Dict[model]["y"], axis=0).tolist()
        avgs[model]["stds_y"] = np.std(Cos_Sim_Dict[model]["y"], axis=0).tolist()
        # pdb.set_trace()

    # save processed data
    f = open(saveDR+saveName+".json", "w")
    json.dump(avgs, f, sort_keys=True, indent=2)

    # step 3: plot in bar chart - xdata
    width = 0.25
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    modelList = []
    kernels = Cos_Sim_Dict["kernels"]
    x = 2*np.arange(len(kernels))
    multiplier = 0

    for model in avgs:
        offset = width*multiplier                
        xdata = avgs[model]["avgs_x"]
        if not(np.all(np.isnan(xdata))):
            print("Plotting model: {}".format(model))
            modelList.append(model)
            bars = ax.bar(x+offset, xdata, width, label=model)
        
            multiplier += 1

    numModelsPlotted = len(modelList)
    ax.set_xticks(x+numModelsPlotted/2 * width, kernels)
    ax.legend(loc="lower right", ncols=numModelsPlotted)
    ax.axis([-1, x[-1]+2, 0.98, 1.0])
    ax.set_xlabel("Averaging Kernel Size")
    ax.set_ylabel("Cosine Similarity (ua)")
    # plt.show()    
    plt.savefig(saveDR+saveName+"_ua.png")

    # step 4: plot in bar chart - ydata    
    fig2,ax2 = plt.subplots(1,1,figsize=(10,8))
    modelList = []
    multiplier = 0

    for model in avgs:
        offset = width*multiplier                
        ydata = avgs[model]["avgs_y"]
        if not(np.all(np.isnan(ydata))):
            print("Plotting model: {}".format(model))
            modelList.append(model)
            bars2 = ax2.bar(x+offset, ydata, width, label=model)
        
            multiplier += 1

    numModelsPlotted = len(modelList)
    ax2.set_xticks(x+numModelsPlotted/2 * width, kernels)
    ax2.legend(loc="lower right", ncols=numModelsPlotted)
    ax2.axis([-1, x[-1]+2, 0.98, 1.0])
    ax2.set_xlabel("Averaging Kernel Size")
    ax2.set_ylabel("Cosine Similarity (va)")
    # plt.show()    
    plt.savefig(saveDR+saveName+"_va.png")

    return

def main(args):
    saveName = args.fname+"_{}imgs".format(args.numImgs)
    if args.step == 0:
        exp_cos_sim(kernel=args.kernel, numImgs=args.numImgs, fname=saveName)
    elif args.step == 1:
        process_cos_sim(saveName=saveName, loadname=args.loadname)
    else:
        print("Invalid flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1) # pass 0 for computing or 1 for post-processing
    parser.add_argument("--numImgs", type=int, default=100)
    parser.add_argument("--loadname", default="cos_sim")
    parser.add_argument("--fname", default="cos_sim")
    parser.add_argument("--kernel", type=int, default=1)
    args = parser.parse_args()
    main(args)
    
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

def exp_cos_sim(kernels=[1, 5, 10, 20], numImgs=10):
    """
    Experiment Physics Metrics: Performs cosine similarity for different averaging amounts
    and computes the energy spectra for the resulting turbulence values.    
    
    """

    # set up saving and results storage
    saveDR = "./physics_metrics/"
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

        # step two: iterate cosine_similarity over kernels

        numKernels = len(kernels)
        
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
        

        print("Image {} : {}".format(ii+1,numImgs))
        for jj in np.arange(numKernels):            
            kernel = kernels[jj]
            print("    kernel: {}".format(kernel))
            
            # BICUBIC
            overlaps,_,_ = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_bi[0,:,:,:], avgKernel=kernel)
            bi_results_x.append(overlaps[0,0])
            bi_results_y.append(overlaps[0,1])

            # RIDGE REGRESSION
            overlaps,_,_ = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_rr[0,:,:,:], avgKernel=kernel)
            rr_results_x.append(overlaps[0,0])
            rr_results_y.append(overlaps[0,1])

            # RANDOM FOREST
            overlaps,_,_ = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_rf[0,:,:,:], avgKernel=kernel)
            rf_results_x.append(overlaps[0,0])
            rf_results_y.append(overlaps[0,1])

            # SR3 REG
            overlaps,_,_ = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_reg_sr3[0,:,:,:], avgKernel=kernel)
            sr3_reg_results_x.append(overlaps[0,0])
            sr3_reg_results_y.append(overlaps[0,1])

            # SR3 DIFF.
            overlaps,_,_ = cos_similarity(ref_patch=curr_label[0,:,:,:], HR_patch=pred_dif_sr3[0,:,:,:], avgKernel=kernel)
            sr3_dif_results_x.append(overlaps[0,0])
            sr3_dif_results_y.append(overlaps[0,1])

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

    Cos_Sim_Dict["kernels"] = kernels

    # step four: write dict to json file in saveDR
    fname = "cos_sim_data.json"
    f = open(saveDR+fname, "w")
    json.dump(Cos_Sim_Dict, f, sort_keys=True, indent=2)
    # pdb.set_trace()
    f.close()

    return


def process_cos_sim():
    # step 1: load the stored json data
    saveDR = "./physics_metrics/"
    fname = "cos_sim_data.json"

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
    fname = "cos_sim_proc.json"
    f = open(saveDR+fname, "w")
    json.dump(avgs, f, sort_keys=True, indent=2)

    # step 3: plot in bar chart - xdata
    width = 0.25
    fig,ax = plt.subplots(1,1,figsize=(10,8))
    modelList = []
    kernels = Cos_Sim_Dict["kernels"]
    x = np.arange(len(kernels))
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
    ax.legend(loc="upper right", ncols=numModelsPlotted)
    ax.axis([-1, x[-1]+1, 0.99, 1.0])
    ax.set_xlabel("Averaging Kernel Size")
    ax.set_ylabel("Cosine Similarity (ua)")
    # plt.show()    
    plt.savefig(saveDR+"cosine_similarity_ua.png")

    # step 4: plot in bar chart - ydata
    width = 0.25
    fig2,ax2 = plt.subplots(1,1,figsize=(10,8))
    modelList = []
    kernels = Cos_Sim_Dict["kernels"]
    x = np.arange(len(kernels))
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
    ax2.legend(loc="upper right", ncols=numModelsPlotted)
    ax2.axis([-1, x[-1]+1, 0.99, 1.0])
    ax2.set_xlabel("Averaging Kernel Size")
    ax2.set_ylabel("Cosine Similarity (va)")
    # plt.show()    
    plt.savefig(saveDR+"cosine_similarity_va.png")

    return

def main(args):
    if args.step == 0:
        exp_cos_sim(kernels=[1,5,10,20], numImgs=args.numImgs)
    elif args.step == 1:
        process_cos_sim()
    else:
        print("Invalid flag.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, default=1) # pass 0 for computing or 1 for post-processing
    parser.add_argument("--numImgs", type=int, default=100)
    args = parser.parse_args()
    main(args)
    



#--------------------------------------------------------------------------------------#
#### EXTRA PHYSICS ANALYSIS CODE:
    
# # pdb.set_trace()
# batch_data = np.load("./dataset/data_matrix.npy")
# batch_labels = np.load("./dataset/label_matrix.npy")

# # pdb.set_trace()

# # set up parameters for tests
# kernels = [1, 5, 10, 20]
# numKernels = len(kernels)
# dataDims = batch_data.shape
# # numBatches = dataDims[0] 
# numBatches = 1 # comment out and uncomment line above for full dataset

# # loop over kernels
# cos_sim = np.zeros((numBatches,2,numKernels))
# # pdb.set_trace()
# for jj in np.arange(numKernels):
#     kernel = kernels[jj]
#     print("Computing for avg. kernel: {}".format(kernel))
#     for ii in np.arange(numBatches):
                    
#         # get current batch data, ua and va (vel. components)
#         temp_data = batch_data[0:numBatches,:,:,:]
#         lr_temp = batch_data[ii,:,:,:]
#         hr_temp = batch_labels[ii,:,:,:]        

#         # do each model separately - also, send in single image per call        
#         bicubic_pred = util.bicubic_interpolation(temp_data)
#         bicubic_pred = bicubic_pred[0]
        
#         # pdb.set_trace()

#         # cosine similarity for check
#         # cos_sim[ii,0,jj], a, b = metrics.cos_similarity(ref_patch=hr_temp[0,:,:], HR_patch=bicubic_pred[0,:,:], avgKernel=kernel) # ua
#         # cos_sim[ii,1,jj], a, b = metrics.cos_similarity(ref_patch=hr_temp[1,:,:], HR_patch=bicubic_pred[1,:,:], avgKernel=kernel) # va

#         # test kinetic energy
#         # kvals, energyRef, energyHR = metrics.kinetic_energy_spectra(ref_patch=hr_temp[0,:,:], HR_patch=bicubic_pred[0,:,:])
#         # pdb.set_trace()

#         # plt.figure()
#         # plt.plot(a[:], b[:])
#         # plt.plot(a, c)
#         # plt.show()

#         pdb.set_trace()

# pdb.set_trace()
# return
#--------------------------------------------------------------------------------------#
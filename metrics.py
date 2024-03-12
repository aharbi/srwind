import os
import numpy as np
import torch
import util
import dataset
import llr
import sr3
import scipy.ndimage as nd
import scipy.fft as spf

from skimage.metrics import structural_similarity as ssim


def peak_signal_to_noise_ratio(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the peak signal-to-noise ratio between a reference high-resolution
    patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The peak signal-to-noise ratio (in dB)
    """
    rmse = 1 / 100 * np.linalg.norm(ref_patch.flatten() - HR_patch.flatten(), 2)
    if rmse != 0:
        psnr = 20 * np.log10(1 / rmse)
        return psnr
    else:
        return float("inf")

def mean_squared_error(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the mean squared error between a reference high-resolution
    patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The mean squared error.
    """
    return (1 / (100 * 100)) * np.linalg.norm(
        ref_patch.flatten() - HR_patch.flatten(), 2
    ) ** 2


def mean_absolute_error(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the mean absolute error between a reference high-resolution
    patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The mean absolute error.
    """
    return (1 / (100 * 100)) * np.linalg.norm(
        ref_patch.flatten() - HR_patch.flatten(), 1
    )


def structural_similarity_index(ref_patch: np.ndarray, HR_patch: np.ndarray):
    """Computes the structural similarity index measure (SSIM) between a reference
    high-resolution patch and a predicted high-resolution patch.

    Args:
        ref_patch (np.ndarray): Reference high-resolution patch.
        HR_patch (np.ndarray): Predicted high-resolution patch.

    Returns:
        float: The structural similarity index.
    """
    return ssim(ref_patch, HR_patch, data_range=1)


def cos_similarity(ref_patch: np.ndarray, HR_patch: np.ndarray, avgKernel = 1):
    """Computes the cosine similarity error between reference patch and predicted HR_patch,
      after averaging over the specified kernel. This is similar to computing the cosine similarity between
      avg-pooled images, but where dimension is not reduced.
      
    Args:
        ref_patch (np.ndarray): reference high-res patch
        HR_patch (np.ndarray): predicted high-res patch
    
    Returns:
        float: value of cosine similarity.
        ndarray: filtered (averaged) reference patch
        ndarray: filtered (averaged) high-res patch
    """

    def filterAvg(subArray):
        # input: array of dimension 1 x n*d, where n and d are the subarray dimensions passed into scipy filter.
        # output: average over all elements in subArray
                
        aVec = np.flatten(subArray)
        return np.mean(aVec)


    # step 1: get scipy generic filter
    ref_filt = nd.generic_filter(input=ref_patch, function=filterAvg, size=avgKernel, mode="wrap")
    hr_filt = nd.generic_filter(input=HR_patch, function=filterAvg, size=avgKernel, mode="wrap")

    # step 2: get cos similarity
    num = ref_filt.dot(hr_filt)
    den = np.linalg.norm(ref_filt)*np.linalg.norm(hr_filt)

    return num/den, ref_filt, hr_filt


def kinetic_energy_spectra(ref_patch: np.ndarray, HR_patch: np.ndarray):
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
    xdims = ref_patch.shape[1] # returns pixels along x-axis
    ydims = ref_patch.shape[0] # gets number of rows in patch

    df = 1/xdims
    freqs = spf.fftfreq(n = xdims, d = df)  # generate frequency domain sample points [cycles/pixel]

    # loop over slices and get averaged ffts
    ref_fft = hr_fft = np.zeros((1,xdims))
    ref_psd = hr_psd = np.zeros((1,xdims))
    
    for ii in np.arange(ydims):
        tempRef = spf.fft(ref_patch[ii,:])
        tempHR = spf.fft(HR_patch[ii,:])

        ref_fft += tempRef
        hr_fft += tempHR

        ref_psd += np.square(np.abs(tempRef))
        hr_psd += np.square(np.abs(tempHR))
    
    ref_fft_avg = ref_fft/ydims
    hr_fft_avg = hr_fft/ydims

    ref_psd_avg = ref_psd/ydims
    hr_psd_avg = hr_psd/ydims

    # shift psds and freqs
    ref_fft_shifted = spf.fftshift(ref_fft_avg)
    hr_fft_shifted = spf.fftshift(hr_fft_avg)
    ref_psd_shifted = spf.fftshift(ref_psd_avg)
    hr_psd_shifted = spf.fftshift(hr_psd_avg)
    freqs_shifted = spf.fftshift(freqs)

     # return everything
    return freqs_shifted, ref_psd_shifted, hr_psd_shifted


def compute_physics_metrics(ref_patches: np.ndarray, HR_patches: np.ndarray, avg_kernel = 5):
    """Same as "compute_metrics" but specifically for cosine overlap and kinetic energy spectra.
    Note that we compute the cosine similarity over an "average" window. This corresponds to determining
    similarity of laminar flow. We then compute the PSD of each image with this laminar component
    subtracted out.

    Args:
        ref_patches: array of size (batch, 2, 100, 100), containing reference patches.
        hr_patches: array of size (batch, 2, 100, 100), containing high-res inferred patches.
        avg_kernel: square dimension of the kernel to average over for cosine similarity.

    Returns:
        np.ndarray: (batch, 2) cosine similarity for batch images, with 2 cols corresponding 
                    to similarity for each x,y wind component.
        
        np.ndarray: (batch, 2, 100) reference PSD spectra for each image (1 row per image), with 2
                    cols representing x,y data and 100 depth representing each value of the PSD.
        
        np.ndarray: (batch, 2, 100) high res PSD spectra for each hr image (1 row per image), with 2
                    cols representing x,y data and 100 depth representing each value of the PSD.
        
        np.ndarray: (1,100) vector of frequency components for PSD data.        
    """

    # initialize output arrays
    dims = ref_patches.size
    numBatch = dims[0]
    numComponents = dims[1]
    numPts = dims[2]

    cos_sim = np.zeros((numBatch, numComponents))
    ref_psds = hr_psds = np.zeros((numBatch, numComponents, numPts))
    
    # loop over all images in batch
    for ii in np.arange(numBatch):
        
        # get full patch
        ref_patch_x = ref_patches[ii,0,:,:]
        ref_patch_y = ref_patches[ii,1,:,:]
        hr_patch_x = HR_patches[ii,0,:,:]
        hr_patch_y = HR_patches[ii,1,:,:]

        # compute cosine similarities
        cos_sim[ii,0], ref_lam_x, hr_lam_x = cos_similarity(ref_patch=ref_patch_x, HR_patch=hr_patch_x, avgKernel=avg_kernel)
        cos_sim[ii,1], ref_lam_y, hr_lam_y = cos_similarity(ref_patch=ref_patch_y, HR_patch=hr_patch_y, avgKernel=avg_kernel)

        # subtract out the averaged patches to get turbulence patches - dims of averaged patches should match originals
        ref_turb_x = ref_patch_x-ref_lam_x
        ref_turb_y = ref_patch_y-ref_lam_y
        hr_turb_x = hr_patch_x-hr_lam_x
        hr_turb_y = hr_patch_y-hr_lam_y

        # compute PSDs from turbulence patches -- frequencies are the same for every image
        tempFreqs, ref_psds[ii,0,:], hr_psds[ii,0,:] = kinetic_energy_spectra(ref_patch=ref_turb_x, HR_patch=hr_turb_x)
        if ii == 1:
            freqs_psd = tempFreqs
        
        tempFreqs, ref_psds[ii,1,:], hr_psds[ii,1,:] = kinetic_energy_spectra(ref_patch=ref_turb_y, HR_patch=hr_turb_y)

    # now return final filled matrices
    return cos_sim, ref_psds, hr_psds, freqs_psd

        

def compute_metrics(ref_patches: np.ndarray, HR_patches: np.ndarray):
    """Computes the performance metrics per patch given the reference HR patches and
    the predicted HR patches.

    Args:
        ref_patches (np.ndarray): Array of size (batch_size, 2, 100, 100)
        HR_patches (np.ndarray): Array of size (batch_size, 2, 100, 100)

    Returns:
        np.ndarray: computed metrics as an array with batch_size rows and
        two columns representing the wind componenets with a depth of four
        representing the computed metrics (PSNR, MSE, MAE, SSIM).
    """

    psnr_log_ua = []
    psnr_log_va = []

    mse_log_ua = []
    mse_log_va = []

    mae_log_ua = []
    mae_log_va = []

    ssim_log_ua = []
    ssim_log_va = []

    n = ref_patches.shape[0]
    for i in range(n):
        psnr_log_ua.append(
            peak_signal_to_noise_ratio(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )
        mse_log_ua.append(
            mean_squared_error(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )
        mae_log_ua.append(
            mean_absolute_error(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )
        ssim_log_ua.append(
            structural_similarity_index(ref_patches[i, 0, :, :], HR_patches[i, 0, :, :])
        )

        psnr_log_va.append(
            peak_signal_to_noise_ratio(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )
        mse_log_va.append(
            mean_squared_error(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )
        mae_log_va.append(
            mean_absolute_error(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )
        ssim_log_va.append(
            structural_similarity_index(ref_patches[i, 1, :, :], HR_patches[i, 1, :, :])
        )

    metrics_array = np.zeros((n, 2, 4))

    metrics_array[:, 0, 0] = psnr_log_ua
    metrics_array[:, 0, 1] = mse_log_ua
    metrics_array[:, 0, 2] = mae_log_ua
    metrics_array[:, 0, 3] = ssim_log_ua

    metrics_array[:, 1, 0] = psnr_log_va
    metrics_array[:, 1, 1] = mse_log_va
    metrics_array[:, 1, 2] = mae_log_va
    metrics_array[:, 1, 3] = ssim_log_va

    return metrics_array


def compute_metrics_llr(
    path: str,
    model_path_ua,
    pca_path_ua,
    scaler_path_ua,
    model_path_va,
    pca_path_va,
    scaler_path_va,
    window_size,
    stride,
):

    file_names = os.listdir(path)
    n = len(file_names)

    metrics_array = np.zeros((256 * n, 2, 4))

    for i, file_name in enumerate(file_names):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        prediction_lr = llr.predict(
            data_matrix=current_data_matrix,
            model_path_ua=model_path_ua,
            pca_path_ua=pca_path_ua,
            scaler_path_ua=scaler_path_ua,
            model_path_va=model_path_va,
            pca_path_va=pca_path_va,
            scaler_path_va=scaler_path_va,
            window_size=window_size,
            stride=stride,
        )

        metrics_array[i * 256 : i * 256 + 256, :, :] = compute_metrics(
            current_label_matrix, prediction_lr
        )

        print("Current Iteration (LLR): {} / {}".format(i, n))

    return metrics_array


def compute_metrics_regression_sr3(
    path: str,
    model_path: str,
    device: str = "cuda",
    num_features: int = 256,
):

    sr3_model = sr3.RegressionSR3(
        device=device,
        num_features=num_features,
        model_path=model_path,
    )

    file_names = os.listdir(path)
    n = len(file_names)

    metrics_array = np.zeros((256 * n, 2, 4))

    for i, file_name in enumerate(file_names):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        current_data_matrix = util.bicubic_interpolation(current_data_matrix)
        current_data_matrix = current_data_matrix.astype(np.float32)

        x = torch.from_numpy(current_data_matrix)

        prediction_sr3 = sr3_model.inference(x).detach().numpy()

        metrics_array[i * 256 : i * 256 + 256, :, :] = compute_metrics(
            current_label_matrix, prediction_sr3
        )

        print("Current Iteration (SR3): {} / {}".format(i, n))

    return metrics_array


def compute_metrics_diffusion_sr3(
    path: str,
    model_path: str,
    device: str = "cuda",
    T: int = 500,
    num_features: int = 256,
):

    sr3_model = sr3.DiffusionSR3(
        device=device,
        T=T,
        num_features=num_features,
        model_path=model_path,
    )

    file_names = os.listdir(path)
    sample_size = 2

    metrics_array = np.zeros((256 * sample_size, 2, 4))

    for i, file_name in enumerate(file_names[:sample_size]):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        current_data_matrix = util.bicubic_interpolation(current_data_matrix)
        current_data_matrix = current_data_matrix.astype(np.float32)

        for j in range(0, 256, 2):
            x = torch.from_numpy(current_data_matrix[j : j + 2])

            prediction_sr3 = sr3_model.inference(x).detach().numpy()

            metrics_array[i * 256 + j : i * 256 + j + 2, :, :] = compute_metrics(
                current_label_matrix[j : j + 2], prediction_sr3
            )

            print("Current Sub-iteration: {} / {}".format(j, 256))

        print("Current Iteration (SR3): {} / {}".format(i, sample_size))

    return metrics_array


def compute_metrics_bicubic(path: str):

    file_names = os.listdir(path)
    n = len(file_names)

    metrics_array = np.zeros((256 * n, 2, 4))

    for i, file_name in enumerate(file_names):
        current_data_matrix, current_label_matrix = dataset.create_single_file_dataset(
            os.path.join(path, file_name)
        )

        prediction_bi = util.bicubic_interpolation(current_data_matrix)

        metrics_array[i * 256 : i * 256 + 256, :, :] = compute_metrics(
            current_label_matrix, prediction_bi
        )

        print("Current Iteration (Bicubic): {} / {}".format(i, n))

    return metrics_array

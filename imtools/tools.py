__author__ = 'Ryba'

import glob


import itertools


import os


import sys


from collections import namedtuple



import matplotlib.pyplot as plt


import numpy as np


import scipy.ndimage.filters as scindifil


import scipy.ndimage.interpolation as scindiint


import scipy.ndimage.measurements as scindimea


import scipy.ndimage.morphology as scindimor


import scipy.signal as scisig


import scipy.stats as scista


import skimage.color as skicol


import skimage.exposure as skiexp


import skimage.feature as skifea


import skimage.filters as skifil


import skimage.io as skiio


import skimage.measure as skimea


import skimage.morphology as skimor


import skimage.restoration as skires


import skimage.segmentation as skiseg


from matplotlib.patches import Ellipse


from mpl_toolkits.axes_grid1 import make_axes_locatable


from skimage.segmentation import mark_boundaries


from sklearn.cluster import MeanShift, estimate_bandwidth

try:
    import cPickle as pickle
except e:
    import pickle

import gzip
import warnings

try:
    import data_viewers
except ImportError:
    pass
#     if os.path.exists('../data_viewers/')
#     sys.path.append('../data_viewers/')
# from dataviewers.seg_viewer import SegViewer
#     import Viewer_3D


# sys.path.append('../seg_viewer/')
# from seg_viewer import SegViewer

#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def get_seeds(im, minT=0.95, maxT=1.05, minInt=0, maxInt=255, debug=False):
    vals = im[np.where(np.logical_and(im>=minInt, im<=maxInt))]
    hist, bins = skiexp.histogram(vals)
    max_peakIdx = hist.argmax()

    minT *= bins[max_peakIdx]
    maxT *= bins[max_peakIdx]
    histTIdxs = (bins >= minT) * (bins <= maxT)
    histTIdxs = np.nonzero(histTIdxs)[0]
    class1TMin = minT
    class1TMax = maxT

    seed_mask = np.where( (im >= class1TMin) * (im <= class1TMax), 1, 0)

    if debug:
        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
        plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
        plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
        plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
        plt.title('Image histogram and its class1 = maximal peak (red dot) +/- minT/maxT % of its density (red lines).')
        plt.show()

    #minT *= hist[max_peakIdx]
    #maxT *= hist[max_peakIdx]
    #histTIdxs = (hist >= minT) * (hist <= maxT)
    #histTIdxs = np.nonzero(histTIdxs)[0]
    #histTIdxs = histTIdxs.astype(np.int)minT *= hist[max_peakIdx]
    #class1TMin = bins[histTIdxs[0]]
    #class1TMax = bins[histTIdxs[-1]

    #if debug:
    #    plt.figure()
    #    plt.plot(bins, hist)
    #    plt.hold(True)
    #
    #    plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
    #    plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
    #    plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
    #    plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
    #    plt.title('Image histogram and its class1 = maximal peak (red dot) +/- minT/maxT % of its density (red lines).')
    #    plt.show()

    return seed_mask, class1TMin, class1TMax


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def seeds2superpixels(seed_mask, superpixels, debug=False, im=None):
    seeds = np.argwhere(seed_mask)
    superseeds = np.zeros_like(seed_mask)

    for s in seeds:
        label = superpixels[s[0], s[1]]
        superseeds = np.where(superpixels==label, 1, superseeds)

    if debug:
        plt.figure(), plt.gray()
        plt.subplot(121), plt.imshow(im), plt.hold(True), plt.plot(seeds[:,1], seeds[:,0], 'ro'), plt.axis('image')
        plt.subplot(122), plt.imshow(im), plt.hold(True), plt.plot(seeds[:,1], seeds[:,0], 'ro'),
        plt.imshow(mark_boundaries(im, superseeds, color=(1,0,0))), plt.axis('image')
        plt.show()

    return superseeds


#----------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------
def intensity_range2superpixels(im, superpixels, intMinT=0.95, intMaxT=1.05, debug=False, intMin=0, intMax=255):#, fromInt=0, toInt=255):

    superseeds = np.zeros_like(superpixels)

    #if not intMin and not intMax:
    #    hist, bins = skexp.histogram(im)
    #
    #    #zeroing values that are lower/higher than fromInt/toInt
    #    toLow = np.where(bins < fromInt)
    #    hist[toLow] = 0
    #    toHigh = np.where(bins > toInt)
    #    hist[toHigh] = 0
    #
    #    max_peakIdx = hist.argmax()
    #    intMin = intMinT * bins[max_peakIdx]
    #    intMax = intMaxT * bins[max_peakIdx]

    sp_means = np.zeros(superpixels.max()+1)
    for sp in range(superpixels.max()+1):
        values = im[np.where(superpixels==sp)]
        mean = np.mean(values)
        sp_means[sp] = mean

    idxs = np.argwhere(np.logical_and(sp_means>=intMin, sp_means<=intMax))
    for i in idxs:
        superseeds = np.where(superpixels==i[0], 1, superseeds)

    if debug:
        plt.figure(), plt.gray()
        plt.imshow(im), plt.hold(True), plt.imshow(mark_boundaries(im, superseeds, color=(1,0,0)))
        plt.axis('image')
        plt.show()

    return superseeds


def show_slice(data, segmentation=None, lesions=None, win_l=50, win_w=350, windowing=False, show='True'):
    if windowing:
        vmin = win_l - win_w / 2
        vmax = win_l + win_w / 2
    else:
        vmin = data.min()
        vmax = data.max()

    plt.figure()
    plt.gray()
    plt.imshow(data, interpolation='nearest', vmin=vmin, vmax=vmax)

    if segmentation is not None:
        plt.hold(True)
        contours = skimea.find_contours(segmentation, 1)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)

    if lesions is not None:
        plt.hold(True)
        contours = skimea.find_contours(lesions, 1)
        for contour in contours:
            plt.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)

    plt.axis('image')

    if show:
        plt.show()


def change_slice_index(data):
    n_slices = data.shape[2]
    data_reshaped = np.zeros(np.hstack((data.shape[2], data.shape[0], data.shape[1])))
    for i in range(n_slices):
        data_reshaped[i, :, :] = data[:, :, i]
    return data_reshaped


def read_data(dcmdir, indices=None, wildcard='*.dcm', type=np.int16):
    import dicom



    dcmlist = []
    for infile in glob.glob(os.path.join(dcmdir, wildcard)):
        dcmlist.append(infile)

    if indices == None:
        indices = range(len(dcmlist))

    data3d = []
    for i in range(len(indices)):
        ind = indices[i]
        onefile = dcmlist[ind]
        if wildcard == '*.dcm':
            data = dicom.read_file(onefile)
            data2d = data.pixel_array
            try:
                data2d = (np.float(data.RescaleSlope) * data2d) + np.float(data.RescaleIntercept)
            except:
                print('problem with RescaleSlope and RescaleIntercept')
        else:
            # data2d = cv2.imread(onefile, 0)
            data2d = skiio.imread(onefile, as_grey=True)

        if len(data3d) == 0:
            shp2 = data2d.shape
            data3d = np.zeros([shp2[0], shp2[1], len(indices)], dtype=type)

        data3d[:,:,i] = data2d

    #need to reshape data to have slice index (ndim==3)
    if data3d.ndim == 2:
        data3d.resize(np.hstack((data3d.shape,1)))

    return data3d


def windowing(data, level=50, width=350, sub1024=False, sliceId=2, out_range=(0, 255)):
    #srovnani na standardni skalu = odecteni 1024HU
    if sub1024:
        data -= 1024

    #zjisteni minimalni a maximalni density
    minHU = level - width / 2
    maxHU = level + width / 2

    if data.ndim == 3:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                #rescalovani intenzity tak, aby skala <minHU, maxHU> odpovidala intervalu <0,255>
                data[:, :, idx] = skiexp.rescale_intensity(data[:, :, idx], in_range=(minHU, maxHU), out_range=(0, 255))
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                #rescalovani intenzity tak, aby skala <minHU, maxHU> odpovidala intervalu <0,255>
                data[idx, :, :] = skiexp.rescale_intensity(data[idx, :, :], in_range=(minHU, maxHU), out_range=(0, 255))
    else:
        data = skiexp.rescale_intensity(data, in_range=(minHU, maxHU), out_range=out_range)

    return data.astype(np.uint8)


def smoothing(data, d=10, sigmaColor=10, sigmaSpace=10, sliceId=2):
    import cv2


    if data.ndim == 3:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                data[:, :, idx] = cv2.bilateralFilter(data[:, :, idx], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                data[idx, :, :] = cv2.bilateralFilter(data[idx, :, :], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    else:
        if data.dtype.type == np.float64:
            # data = skiexp.rescale_intensity(data, in_range=(0, 1), out_range=(0, 255)).astype(np.uint8)
            data = data.astype(np.float32)
        data = cv2.bilateralFilter(data, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    return data


def smoothing_bilateral(data, sigma_space=15, sigma_color=0.05, pseudo_3D='True', sliceId=2):
    if data.ndim == 3 and pseudo_3D:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                temp = skifil.denoise_bilateral(data[:, :, idx], sigma_range=sigma_color, sigma_spatial=sigma_space)
                # temp = skires.denoise_bilateral(data[:, :, idx], sigma_range=sigma_color, sigma_spatial=sigma_space)
                data[:, :, idx] = (255 * temp).astype(np.uint8)
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                temp = skifil.denoise_bilateral(data[idx, :, :], sigma_range=sigma_color, sigma_spatial=sigma_space)
                # temp = skires.denoise_bilateral(data[idx, :, :], sigma_range=sigma_color, sigma_spatial=sigma_space)
                data[idx, :, :] = (255 * temp).astype(np.uint8)
    else:
        data = skifil.denoise_bilateral(data, sigma_range=sigma_color, sigma_spatial=sigma_space)
        # data = skires.denoise_bilateral(data, sigma_range=sigma_color, sigma_spatial=sigma_space)
        data = (255 * data).astype(np.uint8)
    return data


def smoothing_tv(data, weight=0.1, pseudo_3D=True, multichannel=False, sliceId=2):
    if data.ndim == 3 and pseudo_3D:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                # temp = skifil.denoise_tv_chambolle(data[:, :, idx], weight=weight, multichannel=multichannel)
                temp = skires.denoise_tv_chambolle(data[:, :, idx], weight=weight, multichannel=multichannel)
                data[:, :, idx] = (255 * temp).astype(np.uint8)
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                # temp = skifil.denoise_tv_chambolle(data[idx, :, :], weight=weight, multichannel=multichannel)
                temp = skires.denoise_tv_chambolle(data[idx, :, :], weight=weight, multichannel=multichannel)
                data[idx, :, :] = (255 * temp).astype(np.uint8)
    else:
        # data = skifil.denoise_tv_chambolle(data, weight=weight, multichannel=False)
        data = skires.denoise_tv_chambolle(data, weight=weight, multichannel=False)
        data = (255 * data).astype(np.uint8)
    return data


def smoothing_gauss(data, sigma=1, pseudo_3D='True', sliceId=2):
    if data.ndim == 3 and pseudo_3D:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                temp = skifil.gaussian_filter(data[:, :, idx], sigma=sigma)
                data[:, :, idx] = (255 * temp).astype(np.uint8)
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                temp = skifil.gaussian_filter(data[idx, :, :], sigma=sigma)
                data[idx, :, :] = (255 * temp).astype(np.uint8)
    else:
        data = skifil.gaussian_filter(data, sigma=sigma)
        data = (255 * data).astype(np.uint8)
    return data


def analyse_histogram(data, roi=None, dens_min=20, dens_max=255, minT=0.8, maxT=1.2, show=False, show_now=True):
    if roi == None:
        #roi = np.ones(data.shape, dtype=np.bool)
        roi = np.logical_and(data >= dens_min, data <= dens_max)

    voxels = data[np.nonzero(roi)]
    hist, bins = skiexp.histogram(voxels)
    max_peakIdx = hist.argmax()

    minT = minT * hist[max_peakIdx]
    maxT = maxT * hist[max_peakIdx]
    histTIdxs = (hist >= minT) * (hist <= maxT)
    histTIdxs = np.nonzero(histTIdxs)[0]
    histTIdxs = histTIdxs.astype(np.int)

    class1TMin = bins[histTIdxs[0]]
    class1TMax = bins[histTIdxs[-1]]

    main = data * (roi > 0)
    class1 = np.where((main >= class1TMin) * (main <= class1TMax), 1, 0)

    if show:
        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
        plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
        plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
        plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
        plt.title('Histogram and class1 = max peak (red dot) +-5% of its density (red lines).')
        if show_now:
            plt.show()

    return class1


def dominant_class(data, roi=None, dens_min=0, dens_max=255, peakT=0.8, show=False, show_now=True):
    if roi is None:
        #roi = np.ones(data.shape, dtype=np.bool)
        if isinstance(data.dtype, float):
            dens_min /= 255.
            dens_max /= 255.
        roi = np.logical_and(data >= dens_min, data <= dens_max)

    voxels = data[np.nonzero(roi)]
    hist, bins = skiexp.histogram(voxels)

    hist2 = hist_smoothing(bins, hist, sigma=10)

    # plt.figure()
    # plt.fill(bins, hist, 'b', bins, hist2, 'r', alpha=0.7)
    # plt.show()

    hist = hist2.copy()

    max_peak = hist.max()
    max_peak_idx = hist.argmax()

    l_idx = max_peak_idx
    while (hist[l_idx] > (max_peak * peakT)) and (l_idx > 0):
        l_idx -= 1

    r_idx = max_peak_idx
    while (hist[r_idx] > (max_peak * peakT)) and (r_idx < len(hist) - 1):
        r_idx += 1

    dom_l = bins[l_idx]
    dom_r = bins[r_idx]

    main = data * (roi > 0)
    class1 = np.where((main >= dom_l) * (main <= dom_r), 1, 0)

    # std = data[np.nonzero(class1)].std()
    std = 1
    rv = scista.norm(loc=bins[max_peak_idx], scale=std)

    if show:
        plt.figure()
        plt.plot(bins, hist)
        plt.fill_between(bins, hist, color='b')
        plt.hold(True)

        # pdf = rv.pdf(bins)
        # plt.figure()
        # plt.plot(bins, pdf * max_peak / pdf.max(), 'm')
        # plt.show()

        plt.plot(bins[max_peak_idx], hist[max_peak_idx], 'ro', markersize=10)
        plt.plot([bins[l_idx], bins[l_idx]], [0, hist[max_peak_idx]], 'r-', linewidth=4)
        plt.plot([bins[r_idx], bins[r_idx]], [0, hist[max_peak_idx]], 'r-', linewidth=4)
        plt.plot(bins[l_idx], hist[l_idx], 'rx', markersize=10, markeredgewidth=2)
        plt.plot(bins[r_idx], hist[r_idx], 'rx', markersize=10, markeredgewidth=2)
        plt.title('Histogram and dominant_class.')
        if show_now:
            plt.show()

    return class1, rv


def intensity_probability(data, std=20, mask=None, dens_min=10, dens_max=255):
    if mask is None:
        # roi = np.logical_and(data >= dens_min, data <= dens_max)
        roi = np.ones(data.shape, dtype=np.bool)
    voxels = data[np.nonzero(mask)]
    hist, bins = skiexp.histogram(voxels)

    #zeroing histogram outside interval <dens_min, dens_max>
    hist[:dens_min] = 0
    hist[dens_max:] = 0

    max_id = hist.argmax()
    mu = round(bins[max_id])

    prb = scista.norm(loc=mu, scale=std)

    print('main pdf: mu = %i, std = %i' % (mu, std))

    # plt.figure()
    # plt.plot(bins, hist)
    # plt.hold(True)
    # plt.plot(mu, hist[max_id], 'ro')
    # plt.show()

    probs_L = prb.pdf(voxels)
    probs = np.zeros(data.shape)

    coords = np.argwhere(roi)
    n_elems = coords.shape[0]
    for i in range(n_elems):
        if data.ndim == 3:
            probs[coords[i,0], coords[i,1], coords[i,2]] = probs_L[i]
        else:
            probs[coords[i,0], coords[i,1]] = probs_L[i]

    return probs


def get_zunics_compactness(obj):
    if obj.ndim == 2:
        obj = np.expand_dims(obj, 0)
    m000 = obj.sum()
    m200 = get_central_moment(obj, 2, 0, 0)
    m020 = get_central_moment(obj, 0, 2, 0)
    m002 = get_central_moment(obj, 0, 0, 2)
    term1 = (3**(5./3)) / (5 * (4*np.pi)**(2./3))
    term2 = m000**(5./3) / (m200 + m020 + m002)
    K = term1 * term2
    return K


def get_central_moment(obj, p, q, r):
    elems = np.argwhere(obj)
    m000 = obj.sum()
    m100 = (elems[:,0]).sum()
    m010 = (elems[:,1]).sum()
    m001 = (elems[:,2]).sum()
    xc = m100 / m000
    yc = m010 / m000
    zc = m001 / m000

    mom = 0
    for el in elems:
        mom += (el[0] - xc)**p + (el[1] - yc)**q + (el[2] - zc)**r

    return mom


def compactness(obj):
    border = (obj - skimor.binary_erosion(obj, np.ones((3, 3)))).sum()
    area = obj.sum()
    comp = float(border ** 2) / area

    return comp


def opening3D(data, selem=skimor.disk(3), sliceId=0):
    if sliceId == 0:
        for i in range(data.shape[0]):
            data[i,:,:] = skimor.binary_opening(data[i,:,:], selem)
    elif sliceId == 2:
        for i in range(data.shape[2]):
            data[:,:,i] = skimor.binary_opening(data[:,:,i], selem)
    return data


def closing3D(data, selem=skimor.disk(3), slicewise=False, sliceId=0):
    if slicewise:
        if sliceId == 0:
            for i in range(data.shape[0]):
                data[i, :, :] = skimor.binary_closing(data[i, :, :], selem)
        elif sliceId == 2:
            for i in range(data.shape[2]):
                data[:, :, i] = skimor.binary_closing(data[:, :, i], selem)
    else:
        data = scindimor.binary_closing(data, selem)
    return data


def eroding3D(data, selem=None, selem_size=3, slicewise=False, sliceId=0):
    # if selem is None:
    #     if len(data.shape) == 3:
    #         selem = np.ones((selem_size, selem_size, selem_size))
    #     else:
    #         selem = skimor.disk(selem_size)
    # if slicewise:
    #     if sliceId == 0:
    #         for i in range(data.shape[0]):
    #             data[i, :, :] = skimor.binary_erosion(data[i, :, :], selem)
    #     elif sliceId == 2:
    #         for i in range(data.shape[2]):
    #             data[:, :, i] = skimor.binary_erosion(data[:, :, i], selem)
    # else:
    #     data = scindimor.binary_erosion(data, selem)
    data = morph_ND(data, 'erosion', selem, selem_size, slicewise, sliceId)
    return data


def morph_ND(data, method, selem=None, selem_rad=3, slicewise=True, sliceId=0):
    if method == 'erosion':
        morph_func = scindimor.binary_erosion
    elif method == 'dilation':
        morph_func = scindimor.binary_dilation
    elif method == 'opening':
        morph_func = scindimor.binary_opening
    elif method == 'closing':
        morph_func = scindimor.binary_closing
    else:
        raise ValueError('Wrong morphological operation name.')

    if selem is None:
        selem = np.ones((2 * selem_rad + 1,) * data.ndim)

    if data.ndim == 2:
        data = morph_func(data, selem)
    else:
        if slicewise:
            if sliceId == 0:
                for i in range(data.shape[0]):
                    data[i, :, :] = morph_func(data[i, :, :], selem)
            elif sliceId == 2:
                for i in range(data.shape[2]):
                    data[:, :, i] = morph_func(data[:, :, i], selem)
        else:
            data = morph_func(data, selem)
    return data


def resize3D(data, scale=None, shape=None, sliceId=2, method='cv2'):
    import cv2


    if data.ndim == 2:
        if shape is not None:
            new_data = cv2._resize_if_required(data.astype(np.uint8), shape, 0, 0, interpolation=cv2.INTER_NEAREST)
        elif method == 'cv2':
            if data.dtype == np.bool:
                data = data.astype(np.uint8)
            new_data = cv2._resize_if_required(data, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        else:
            new_data = scindiint.zoom(data, scale)

    else:
        if sliceId == 2:
            n_slices = data.shape[2]
            # new_shape = cv2.resize(data[:,:,0], None, fx=scale, fy=scale).shape
            new_shape = scindiint.zoom(data[:,:,0], scale).shape
            new_data = np.zeros(np.hstack((new_shape,n_slices)), dtype=np.int)
            for i in range(n_slices):
                # new_data[:,:,i] = cv2.resize(data[:,:,i], None, fx=scale, fy=scale)
                # new_data[:,:,i] = (255 * skitra.rescale(data[:,:,0], scale)).astype(np.int)
                if method == 'cv2':
                    new_data[:,:,i] = cv2._resize_if_required(data[:, :, i], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                else:
                    new_data[:,:,i] = scindiint.zoom(data[:,:,i], scale)
        elif sliceId == 0:
            n_slices = data.shape[0]
            # new_shape = cv2.resize(data[0,:,:], None, fx=scale, fy=scale).shape
            # new_shape = skitra.rescale(data[0,:,:], scale).shape
            if method == 'cv2':
                if data.dtype == np.bool:
                    data = data.astype(np.uint8)
                new_shape = cv2._resize_if_required(data[0, :, :], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).shape
            else:
                new_shape =  scindiint.zoom(data[0,:,:], scale).shape
            new_data = np.zeros(np.hstack((n_slices, new_shape)), dtype=np.int)
            for i in range(n_slices):
                # new_data[i,:,:] = cv2.resize(data[i,:,:], None, fx=scale, fy=scale)
                # new_data[i,:,:] = (255 * skitra.rescale(data[i,:,:], scale)).astype(np.int)
                if method == 'cv2':
                    new_data[i,:,:] = cv2._resize_if_required(data[i, :, :], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                else:
                    new_data[i,:,:] = scindiint.zoom(data[i,:,:], scale)
    return new_data


def resize_ND(data, scale=None, shape=None, slice_id=0, method='cv2'):
    if shape is None:
        shape = list(data.shape)
    else:
        shape = list(shape)

    if data.ndim == 2:
        data = np.expand_dims(data, 0)
        shape.insert(0, 1)
        expanded = True
    else:
        expanded = False

    if slice_id == 2:
        data = np.swapaxes(np.swapaxes(data, 0, 2), 1, 2)
        shape = [shape[2], shape[0], shape[1]]
        swapped = True
    else:
        swapped = False

    if scale is not None:
        new_slice_shape = cv2._resize_if_required(data[0, ...], (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST).shape
    else:
        new_slice_shape = shape[1:]

    # new_data = np.zeros(np.hstack((data.shape[0], new_slice_shape)), dtype=np.int)
    # data = data.astype(np.uint8)

    new_data = np.zeros(np.hstack((data.shape[0], new_slice_shape)), dtype=data.dtype)
    # data = data.astype(np.uint8)

    for i, im in enumerate(data):
        if scale is not None:
            new_data[i, ...] = cv2._resize_if_required(im, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        elif shape is not None:
            new_data[i, ...] = cv2._resize_if_required(im, (shape[2], shape[1]), interpolation=cv2.INTER_NEAREST)

    if expanded:
        new_data = new_data[0, ...]

    if swapped:
        new_data = np.swapaxes(np.swapaxes(new_data, 0, 2), 1, 2)

    return new_data


def get_overlay(mask, alpha=0.3, color='r'):
    layer = None
    if color == 'r':
        layer = np.dstack((255*mask, np.zeros_like(mask), np.zeros_like(mask), alpha * mask))
    elif color == 'g':
        layer = alpha * np.dstack((np.zeros_like(mask), mask, np.zeros_like(mask)))
    elif color == 'b':
        layer = alpha * np.dstack((np.zeros_like(mask), np.zeros_like(mask), mask))
    elif color == 'c':
        layer = alpha * np.dstack((np.zeros_like(mask), mask, mask))
    elif color == 'm':
        layer = alpha * np.dstack((mask, np.zeros_like(mask), mask))
    elif color == 'y':
        layer = alpha * np.dstack((mask, mask, np.zeros_like(mask)))
    else:
        print('Unknown color, using red as default.')
        layer = alpha * np.dstack((mask, np.zeros_like(mask), np.zeros_like(mask)))
    return layer


def slim_seeds(seeds, sliceId=2):
    slims = np.zeros_like(seeds)
    if sliceId == 0:
        for i in range(seeds.shape[0]):
            layer = seeds[i,:,:]
            labels = skimor.label(layer, neighbors=4, background=0) + 1
            n_labels = labels.max()
            for o in range(1,n_labels+1):
                centroid = np.round(skimea.regionprops(labels == o)[0].centroid)
                slims[i, centroid[0], centroid[1]] = 1
    return slims


def crop_to_bbox(im, mask):
    if im.ndim == 2:
        # obj_rp = skimea.regionprops(mask.astype(np.integer), properties=('BoundingBox'))
        obj_rp = skimea.regionprops(mask.astype(np.integer))
        bbox = obj_rp[0].bbox # minr, minc, maxr, maxc

        bbox = np.array(bbox)
        # okrajove podminky
        # bbox[0] = max(0, bbox[0]-1)
        # bbox[1] = max(0, bbox[1]-1)
        # bbox[2] = min(im.shape[0], bbox[2]+1)
        # bbox[3] = min(im.shape[1], bbox[3]+1)
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(im.shape[0], bbox[2])
        bbox[3] = min(im.shape[1], bbox[3])

        # im = im[bbox[0]-1:bbox[2] + 1, bbox[1]-1:bbox[3] + 1]
        # mask = mask[bbox[0]-1:bbox[2] + 1, bbox[1]-1:bbox[3] + 1]
        im = im[bbox[0]:bbox[2], bbox[1]:bbox[3]]
        mask = mask[bbox[0]:bbox[2], bbox[1]:bbox[3]]

    elif im.ndim == 3:
        coords = np.nonzero(mask)
        s_min = max(0, min(coords[0]))
        s_max = min(im.shape[0], max(coords[0]))
        r_min = max(0, min(coords[1]))
        r_max = min(im.shape[1], max(coords[1]))
        c_min = max(0, min(coords[2]))
        c_max = min(im.shape[2], max(coords[2]))

        # im = im[r_min-1:r_max+1, c_min-1:c_max+1, s_min-1:s_max+1]
        # mask = mask[r_min-1:r_max+1, c_min-1:c_max+1, s_min-1:s_min+1]
        im = im[s_min:s_max + 1, r_min:r_max + 1, c_min:c_max + 1]
        mask = mask[s_min:s_max + 1, r_min:r_max + 1, c_min:c_max + 1]

    return im, mask


def slics_3D(im, pseudo_3D=True, n_segments=100, get_slicewise=False):
    import cv2


    if im.ndim != 3:
        raise Exception('3D image is needed.')

    if not pseudo_3D:
        # need to convert to RGB image
        im_rgb = np.zeros((im.shape[0], im.shape[1], im.shape[2], 3))
        im_rgb[:,:,:,0] = im
        im_rgb[:,:,:,1] = im
        im_rgb[:,:,:,2] = im

        suppxls = skiseg.slic(im_rgb, n_segments=n_segments, spacing=(2,1,1))

    else:
        suppxls = np.zeros(im.shape)
        if get_slicewise:
            suppxls_slicewise = np.zeros(im.shape)
        offset = 0
        for i in range(im.shape[0]):
            suppxl = skiseg.slic(cv2.cvtColor(im[i,:,:], cv2.COLOR_GRAY2RGB), n_segments=n_segments)
            suppxls[i,:,:] = suppxl + offset
            if get_slicewise:
                suppxls_slicewise[i,:,:] = suppxl
            offset = suppxls.max() + 1

    if get_slicewise:
        return suppxls, suppxls_slicewise
    else:
        return suppxls


def get_superpxl_intensities(im, suppxls):
    """Calculates mean intensities of pixels in superpixels
    inputs:
        im ... grayscale image, ndarray [MxN]
        suppxls ... image with suppxls labels, ndarray -same shape as im
    outputs:
        suppxl_intens ... vector with suppxls mean intensities
    """
    n_suppxl = np.int(suppxls.max() + 1)
    suppxl_intens = np.zeros(n_suppxl)

    for i in range(n_suppxl):
        sup = suppxls == i
        vals = im[np.nonzero(sup)]
        try:
            suppxl_intens[i] = np.mean(vals)
        except:
            suppxl_intens[i] = -1

    return suppxl_intens


def suppxl_ints2im(suppxls, suppxl_ints=None, im=None):
    """Replaces superpixel labels with their mean value.
    inputs:
        suppxls ... image with suppxls labels, ndarray
        suppxl_intens ... vector with suppxls mean intensities
        im ... input image
    outputs:
        suppxl_ints_im ... image with suppxls mean intensities, ndarray same shape as suppxls
    """

    suppxl_ints_im = np.zeros(suppxls.shape)

    if suppxl_ints is None and im is not None:
        suppxl_ints = get_superpxl_intensities(im, suppxls)

    for i in np.unique(suppxls):
        sup = suppxls == i
        val = suppxl_ints[i]
        suppxl_ints_im[np.nonzero(sup)] = val

    return suppxl_ints_im


def remove_empty_suppxls(suppxls):
    """Remove empty superpixels. Sometimes there are superpixels(labels), which are empty. To overcome subsequent
    problems, these empty superpixels should be removed.
    inputs:
        suppxls ... image with suppxls labels, ndarray [MxN]-same size as im
    outputs:
        new_supps ... image with suppxls labels, ndarray [MxN]-same size as im, empty superpixel labels are removed
    """
    n_suppxls = np.int(suppxls.max() + 1)
    new_supps = np.zeros(suppxls.shape, dtype=np.integer)
    idx = 0
    for i in range(n_suppxls):
        sup = suppxls == i
        if sup.any():
            new_supps[np.nonzero(sup)] = idx
            idx += 1
    return new_supps


def label_3D(data, class_labels, background=-1):
    # class_labels = np.unique(data[data > background])
    labels = - np.ones(data.shape, dtype=np.int)
    curr_l = 0
    for c in class_labels:
        x = data == c
        labs, n_labels = scindimea.label(x)
        print('labels: ', np.unique(labs))
        # py3DSeedEditor.py3DSeedEditor(labs).show()
        for l in range(n_labels + 1):
            labels = np.where(labs == l, curr_l, labels)
            curr_l += 1
    print('min = %i, max = %i' % (labels.min(), labels.max()))
    return labels


def get_hist_mode(im, mask=None, debug=False):
    if mask is None:
        mask = np.ones(im.shape, dtype=np.bool)
    data = im[np.nonzero(mask)]

    hist, bins = skiexp.histogram(data)
    max_peak_idx = hist.argmax()

    mode = bins[max_peak_idx]

    if debug:
        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        plt.plot(bins[max_peak_idx], hist[max_peak_idx], 'ro')
        plt.title('Histogram of input data with marked mode = %i' % mode)
        plt.show()

    return mode


def smoothing_float_tv(data, weight=0.01, pseudo_3D=True, multichannel=False, sliceId=2):
    if data.ndim == 3 and pseudo_3D:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                in_min = data[:, :, idx].min()
                in_max = data[:, :, idx].max()
                temp = skires.denoise_tv_chambolle(data[:, :, idx], weight=weight, multichannel=multichannel)
                temp = skiexp.rescale_intensity(temp, (temp.min(), temp.max()), (in_min, in_max)).astype(data.dtype)
                data[:, :, idx] = temp
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                in_min = data[idx, :, :].min()
                in_max = data[idx, :, :].max()
                temp = skires.denoise_tv_chambolle(data[idx, :, :], weight=weight, multichannel=multichannel)
                temp = skiexp.rescale_intensity(temp, (temp.min(), temp.max()), (in_min, in_max)).astype(data.dtype)
                data[idx, :, :] = temp
    else:
        in_min = data.min()
        in_max = data.max()
        temp = skires.denoise_tv_chambolle(data, weight=weight, multichannel=False)
        data = skiexp.rescale_intensity(temp, (temp.min(), temp.max()), (in_min, in_max)).astype(data.dtype)
    return data


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = skifea.canny(image, low_threshold=lower, high_threshold=upper)

    # return the edged image
    return edged


def morph_hat(img, strels, type='tophat', show=False, show_now=True):
    if not isinstance(strels, list):
        strels = [strels]
    n_strels = len(strels)

    resps = []
    for strel in strels:
        # resp = cv2.morphologyEx(img, op, strel)
        if type in ['tophat', 'whitehat']:
            # op = cv2.MORPH_TOPHAT
            resp = skimor.white_tophat(img, strel)
        elif type == 'blackhat':
            # op = cv2.MORPH_BLACKHAT
            resp = skimor.black_tophat(img, strel)
        else:
            raise ValueError('Wrong operation type. Only \'tophat\' and \'blackhat\' are allowed.')
        resps.append(resp)

    if show:
        plt.figure()
        plt.imshow(img, 'gray', interpolation='nearest')
        plt.title('input')

        max_resp = np.array(resps).max()
        for i in range(n_strels):
            plt.figure()
            plt.imshow(resps[i], 'gray', interpolation='nearest', vmax=max_resp)
            # plt.imshow(resps[i], 'gray', interpolation='nearest'), plt.colorbar()
            plt.title('{}, strel = {}'.format(type, strels[i].shape))

        if show_now:
            plt.show()

    return resps


def get_status_text(text, iter, max_iter):
    if iter == -1:
        done = '#' * max_iter
        line = '\r' + text + ': [%s]' % done
    else:
        done = '#' * iter
        remain = '-' * (max_iter - iter - 1)
        line = '\r' + text + ': [%s~%s]' % (done, remain)

    return line


def load_pickle_data(fname, slice_idx=-1, return_datap=False):
    ext_list = ('pklz', 'pickle')
    if fname.split('.')[-1] in ext_list:

        try:
            import gzip


            f = gzip.open(fname, 'rb')
            fcontent = f.read()
            f.close()
        except Exception as e:
            f = open(fname, 'rb')
            fcontent = f.read()
            f.close()
        data_dict = pickle.loads(fcontent)

        if return_datap:
            return data_dict

        # data = tools.windowing(data_dict['data3d'], level=params['win_level'], width=params['win_width'])
        data = data_dict['data3d']
        mask = data_dict['segmentation']
        voxel_size = data_dict['voxelsize_mm']

        # TODO: predelat na 3D data
        if slice_idx != -1:
            data = data[slice_idx, :, :]
            mask = mask[slice_idx, :, :]

        return data, mask, voxel_size

    else:
        msg = 'Wrong data type, supported extensions: ', ', '.join(ext_list)
        raise IOError(msg)


def get_bbox(im):
    '''
    Returns bounding box in slicing-friendly format (i-min, i-xmax, j-min, j-max, k-min, k-max).
    The fourth and fifth items are returned only if input image is 3D.
    Parameters
    ----------
    im - input binary image

    Returns
    -------
    indices - (i-min, i-xmax, j-min, j-max, k-min, k-max) in 3D case, (i-min, i-xmax, j-min, j-max) in 2D case.
    '''
    coords = np.nonzero(im)
    inds = []
    for i in coords:
        i_min = i.min()
        i_max = i.max()
        inds.extend((i_min, i_max))

    return inds


def get_subdir_fname(data_fname, subdir, ext='npy', create=False):
    dirs = data_fname.split('/')
    dir = dirs[-1]
    patient_id = dir[8:11]
    if 'venous' in dir or 'ven' in dir:
        phase = 'venous'
    elif 'arterial' in dir or 'art' in dir:
        phase = 'arterial'
    else:
        phase = 'phase_unknown'

    dirname = os.path.join(os.sep.join(dirs[:-1]), subdir)
    if create:
        if not os.path.exists(dirname):
            os.mkdir(dirname)

    # seeds_fname = os.path.join(os.sep.join(dirs[:-1]), 'seeds', 'seeds_%s_%s.npy' % (patient_id, phase))
    subdir_fname = os.path.join(dirname, '%s_%s_%s.%s' % (subdir, patient_id, phase, ext))

    # return seeds_fname
    return subdir_fname


def put_bbox_back(im_o, im_b, bbox=None, mask=None):
    im_o = im_o.copy()
    if bbox is None and mask is None:
        raise ValueError('Bbox or mask must be defined.')

    if mask is not None:
        bbox = get_bbox(mask)

    if len(bbox) == 6:
        im_o[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1, bbox[4]:bbox[5] + 1] = im_b
    else:
        im_o[bbox[0]:bbox[1] + 1, bbox[2]:bbox[3] + 1] = im_b

    return im_o


def save_pickle(data_fname, subdir, data, mask, segs, voxel_size, slab=None):
    segs_fname = get_subdir_fname(data_fname, subdir, ext='pklz', create=True)
    datap = {'data3d': data, 'mask': mask, 'segmentation': segs, 'voxelsize_mm': voxel_size, 'slab': slab}

    f = gzip.open(segs_fname, 'wb', compresslevel=1)
    pickle.dump(datap, f)
    f.close()


def save_figs(data_fname, subdir, data, mask, imgs, ranges=None, cmaps=None):
    dirs = data_fname.split('/')
    dir = dirs[-1]
    patient_id = dir[8:11]
    if 'venous' in dir or 'ven' in dir:
        phase = 'venous'
    elif 'arterial' in dir or 'art' in dir:
        phase = 'arterial'
    else:
        phase = 'phase_unknown'

    # checking directories
    fig_dir = os.path.join(os.sep.join(dirs[:-1]), subdir)
    if not os.path.exists(fig_dir):
        os.mkdir(fig_dir)
    fig_patient_dir = os.path.join(fig_dir, 'figs_%s_%s' % (patient_id, phase))
    if not os.path.exists(fig_patient_dir):
        os.mkdir(fig_patient_dir)

    # saving figures
    if isinstance(imgs, np.ndarray):
        imgs = [imgs]

    fig = plt.figure(figsize=(20, 5))
    for s in range(len(imgs)):
        print(get_status_text('\tSaving figures', iter=s, max_iter=len(imgs)),)
        # print 'Saving figure #%i/%i ...' % (s + 1, len(res)),
        for i, (title, im) in enumerate(imgs[s]):
            plt.subplot(1, len(imgs[s]), i + 1)
            if 'contours' in title:
                plt.imshow(im[0], 'gray', interpolation='nearest'), plt.title(title)
                plt.hold(True)
                for n, contour in enumerate(im[1]):
                    plt.plot(contour[:, 1], contour[:, 0], linewidth=2)
                plt.axis('image')
            elif title == 'input':
                plt.imshow(data[s,:,:], 'gray', interpolation='nearest'), plt.title(title)
            else:
                if cmaps is not None:
                    cmap = cmaps[i]
                else:
                    cmap = 'gray'
                if ranges is not None:
                    plt.imshow(im, cmap=cmap, vmin=ranges[i][0], vmax=ranges[i][1], interpolation='nearest'), plt.title(title)
                else:
                    plt.imshow(im, cmap=cmap, interpolation='nearest'), plt.title(title)

        fig.savefig(os.path.join(fig_patient_dir, 'slice_%i.png' % s))
        fig.clf()
        # print 'done'
    print(get_status_text('\tSaving figures', iter=-1, max_iter=len(imgs)))


def view_segmentation(datap_1, datap_2=None):
    if not os.path.exists('../data_viewers/'):
        print('Package data_viewers not found.')
    else:
        sys.path.append('../data_viewers/')
        from dataviewers.seg_viewer import SegViewer



        from PyQt5 import QtGui, QtWidgets
        app = QtWidgets.QApplication(sys.argv)
        le = SegViewer(datap1=datap_1, datap2=datap_2)
        le.show()
        sys.exit(app.exec_())


def show_3d(data, range=True):
    if not os.path.exists('/home/tomas/projects/data_viewers'):
        print('Package data_viewers not found.')
    else:
        sys.path.append('/home/tomas/projects/data_viewers')
        from dataviewers.viewer_3D import Viewer_3D


        if isinstance(data, tuple):
            # n_data = len(data)
            n_slices = data[0].shape[0]
            n_rows = data[0].shape[1]
            n_cols = sum([x.shape[2] for x in data])
            data_vis = np.zeros((n_slices, n_rows, n_cols))

            # data_vis = []
            # for i in data:
            #     data_vis.append(skiexp.rescale_intensity(i, out_range=np.uint8))
            data = [skiexp.rescale_intensity(x, out_range=np.uint8) for x in data]
            for i in xrange(n_slices):
                slice = []
                for j in data:
                    # slice.append(skiexp.rescale_intensity((j[i, :, :]).astype(np.uint8), out_range=np.uint8))
                    slice.append(j[i, :, :])
                # data_vis[i, :, :] = np.hstack(slice)
                data_vis[i, :, :] = np.hstack(slice)
            # data_vis = np.hstack(data_vis)
        else:
            data_vis = data

        from PyQt5 import QtGui, QtWidgets
        app = QtWidgets.QApplication(sys.argv)
        viewer = Viewer_3D(data_vis, range=True)
        viewer.show()
        app.exec_()
        # sys.exit(app.exec_())


def arange_figs(imgs, tits=None, max_r=3, max_c=5, same_range=False, colorbar=False, show_now=True):
    n_imgs = len(imgs)
    max_imgs = max_r * max_c
    if isinstance(imgs[0], tuple):
        tits = [x[0] for x in imgs]
        imgs = [x[1] for x in imgs]
    else:
        if tits is None:
            tits = [str(x + 1) for x in range(n_imgs)]

    n_rows = int(np.ceil(n_imgs / float(max_c)))
    n_cols = min(n_imgs, max_c)

    if n_imgs > max_imgs:
        imgs_rem = imgs[max_imgs:]
        tits_rem = tits[max_imgs:]
        imgs = imgs[:min(max_imgs, n_imgs)]
        tits = tits[:min(max_imgs, n_imgs)]
        n_rem = n_imgs - max_imgs
    else:
        n_rem = 0

    if same_range:
        vmin = imgs.min()
        vmax = imgs.max()
    fig = plt.figure()
    for i, (im, tit) in enumerate(zip(imgs, tits)):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.imshow(im, 'gray', interpolation='nearest')
        if same_range:
            plt.imshow(im, 'gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        if colorbar:
            plt.colorbar()
        plt.title(tit)

    if n_rem > 0:
        arange_figs(imgs_rem, tits=tits_rem)
    elif show_now:
        plt.show()


def resize(image, width=None, height=None, inter=None):
    import cv2


    if inter is None:
        inter=cv2.INTER_AREA
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2._resize_if_required(image, dim, interpolation=inter)

    # return the resized image
    return resized


def pyramid(image, scale=2, min_size=(30, 30), inter=None):
    """
    Creates generator of image pyramid.
    :param image: input image
    :param scale: factor that controls by how much the image is resized at each layer
    :param min_size: minimum required width and height of the layer
    :return: generator of the image pyramid
    """
    yield image
    import cv2


    if inter is None:
        inter=cv2.INTER_AREA
    # yield the original image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = resize(image, width=w, inter=inter)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < min_size[1] or image.shape[1] < min_size[0]:
            break

        # yield the next image in the pyramid
        yield image


def pyramid_down(image, scale=2, min_size=(30, 30), inter=None, smooth=False):
    if inter is None:
        inter = cv2.INTER_AREA

    w = int(image.shape[1] / scale)
    img = resize(image, width=w, inter=inter)
    if smooth:
        # img = smoothing(image.astype(np.uint8), d=20, sigmaColor=20, sigmaSpace=20).astype(image.dtype)
        img = smoothing(img.astype(np.uint8)).astype(img.dtype)

    if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
        return None
    else:
        range = (image.min(), image.max())
        img = skiexp.rescale_intensity(img, in_range=range, out_range=range)
        return img


def sliding_window(image, step_size, window_size, mask=None, only_whole=True):
    """
    Creates generator of sliding windows.
    :param image: input image
    :param step_size: number of pixels we are going to skip in both the (x, y) direction
    :param window_size: the width and height of the window we are going to extract
    :param mask: region of interest, if None it will slide through the whole image
    :param only_whole: if True - produces only windows of the given window_size
    :return: generator that produce upper left corner of the window, center of the window and the sliding window itself
    """
    if mask is None:
        mask = np.ones(image.shape, dtype=np.bool)
    # slide a window across the image
    for y in xrange(0, image.shape[0], step_size):
        # c_y = y + window_size[1] / 2.
        for x in xrange(0, image.shape[1], step_size):
            # c_x = x + window_size[0] / 2.
            # if c_x < mask.shape[1] and c_y < mask.shape[0] and mask[c_y, c_x]:
            # yield the current window
            end_x = x + window_size[0]
            end_y = y + window_size[1]
            if only_whole and (end_x > image.shape[1] or end_y > image.shape[0]):
                continue
            else:
                mask_out = np.zeros(image.shape, dtype=np.bool)
                mask_out[y:end_y, x:end_x] = True
                yield (x, y, mask_out, image[y:end_y, x:end_x])


def sliding_window_3d(image, step_size, window_size, mask=None, only_whole=True, include_last=False):
    """
    Creates generator of sliding windows.
    :param image: input image
    :param step_size: number of pixels we are going to skip in both the (x, y) direction
    :param window_size: the width and height of the window we are going to extract
    :param mask: region of interest, if None it will slide through the whole image
    :param only_whole: if True - produces only windows of the given window_size
    :return: generator that produce upper left corner of the window, center of the window and the sliding window itself
    """
    if not isinstance(step_size, tuple):
        step_size = (step_size, step_size, step_size)
    if image.ndim == 2:
        image = np.expand_dims(image, 0)
        window_size = (1, window_size[0], window_size[1])
        if mask is not None:
            mask = np.expand_dims(mask, 0)
    if mask is None:
        mask = np.ones(image.shape, dtype=np.bool)
    # slide a window across the image
    for z in xrange(0, image.shape[0], step_size[0]):
        # c_z = z + window_size[0] / 2.
        for y in xrange(0, image.shape[1], step_size[1]):
            # c_y = y + window_size[2] / 2.
            for x in xrange(0, image.shape[2], step_size[2]):
                # c_x = x + window_size[1] / 2.
                # if c_z < mask.shape[0] and c_x < mask.shape[2] and c_y < mask.shape[1] and mask[c_z, c_y, c_x]:
                # yield the current window
                end_x = x + window_size[1]
                end_y = y + window_size[2]
                end_z = z + window_size[0]
                if only_whole and (end_z > image.shape[0] or end_x > image.shape[2] or end_y > image.shape[1]):
                    # if only_whole:
                    continue
                    # elif include_last:
                    #     mask_out = np.zeros(image.shape, dtype=np.bool)
                    #     x = image.shape[2] - window_size[1]
                    #     y = image.shape[1] - window_size[2]
                    #     z = image.shape[0] - window_size[0]
                    #     end_x = image.shape[2]
                    #     end_y = image.shape[1]
                    #     end_z = image.shape[0]
                    #
                    #     mask_out[z:end_z, y:end_y, x:end_x] = True
                    #     yield (x, y, z, mask_out, image[z:end_z, y:end_y, x:end_x])
                else:
                    mask_out = np.zeros(image.shape, dtype=np.bool)
                    mask_out[z:end_z, y:end_y, x:end_x] = True
                    yield (x, y, z, mask_out, image[z:end_z, y:end_y, x:end_x])


def fill_holes(data, slicewise=True, slice_id=0):
    data_o = np.zeros_like(data)
    if data.ndim == 3:
        if slicewise:
            for i in range(data.shape[slice_id]):
                if slice_id == 0:
                    data_o[i, :, :] = scindimor.binary_fill_holes(data[i, :, :])
                    # plt.figure()
                    # plt.subplot(121), plt.imshow(data[i, :, :], 'gray')
                    # plt.subplot(122), plt.imshow(data_o[i, :, :], 'gray')
                    # plt.show()
                elif slice_id == 2:
                    data_o[:, :, i] = scindimor.binary_fill_holes(data[:, :, 1])
    else:
        data_o = scindimor.binary_fill_holes(data)

    return data_o


def fill_holes_watch_borders(mask):
    mask_ex = np.zeros([x + 2 for x in mask.shape], dtype=np.uint8)
    if mask.ndim == 3:
        for i, im in enumerate(mask):
            mask_ex[i + 1, 1:-1, 1:-1] = im
    else:
        mask_ex[1:-1, 1:-1] = mask
    mask_ex = skimor.remove_small_holes(mask_ex, min_size=0.1 * mask.sum(), connectivity=2)
    if mask.ndim == 3:
        mask_ex = mask_ex[1:-1, 1:-1, 1:-1]
    else:
        mask_ex = mask_ex[1:-1, 1:-1]

    return mask_ex


def sigmoid(image, mask=None, a=0.1, c=20, sigm_t = 0.2):
    if mask is None:
        mask = np.ones_like(image)

    im_sigm = (1. / (1 + (np.exp(-a * (image - c))))) * mask
    im_sigm *= (im_sigm > sigm_t)

    return im_sigm


def split_to_tiles(img, columns, rows):
    """
    Split an image into a specified number of tiles.
    Args:
       img (ndarray):  The image to split.
       number_tiles (int):  The number of tiles required.
    Returns:
        Tuple of tiles
    """
    # validate_image(img, number_tiles)

    im_w, im_h = img.shape
    # columns, rows = calc_columns_rows(number_tiles)
    # extras = (columns * rows) - number_tiles
    tile_w, tile_h = int(np.floor(im_w / columns)), int(np.floor(im_h / rows))

    tiles = []
    # number = 1
    for pos_y in range(0, im_h - rows, tile_h):  # -rows for rounding error.
        for pos_x in range(0, im_w - columns, tile_w):  # as above.
            roi = (pos_x, pos_y, pos_x + tile_w, pos_y + tile_h)
            # image = img.crop(area)
            tile = img[roi[1]:roi[3], roi[0]:roi[2]]
            # position = (int(floor(pos_x / tile_w)) + 1,
            #             int(floor(pos_y / tile_h)) + 1)
            # coords = (pos_x, pos_y)
            # tile = Tile(image, number, position, coords)
            tiles.append(tile)
            # number += 1

    return tuple(tiles)


def make_neighborhood_matrix(im, nghood=4, roi=None):
    im = np.array(im, ndmin=3)
    n_slices, n_rows, n_cols = im.shape

    # initialize ROI
    if roi is None:
        roi = np.ones(im.shape, dtype=np.bool)

    # if len(im.shape) == 3:
    #     nslices = im.shape[2]
    # else:
    #     nslices = 1
    npts = n_rows * n_cols * n_slices
    # print 'start'
    if nghood == 8:
        nr = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        nc = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        ns = np.zeros(nghood)
    elif nghood == 4:
        nr = np.array([-1, 0, 0, 1])
        nc = np.array([0, -1, 1, 0])
        ns = np.zeros(nghood, dtype=np.int32)
    elif nghood == 26:
        nr_center = np.array([-1, -1, -1, 0, 0, 1, 1, 1])
        nc_center = np.array([-1, 0, 1, -1, 1, -1, 0, 1])
        nr_border = np.zeros([-1, -1, -1, 0, 0, 0, 1, 1, 1])
        nc_border = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
        nr = np.array(np.hstack((nr_border, nr_center, nr_border)))
        nc = np.array(np.hstack((nc_border, nc_center, nc_border)))
        ns = np.array(np.hstack((-np.ones_like(nr_border), np.zeros_like(nr_center), np.ones_like(nr_border))))
    elif nghood == 6:
        nr_center = np.array([-1, 0, 0, 1])
        nc_center = np.array([0, -1, 1, 0])
        nr_border = np.array([0])
        nc_border = np.array([0])
        nr = np.array(np.hstack((nr_border, nr_center, nr_border)))
        nc = np.array(np.hstack((nc_border, nc_center, nc_border)))
        ns = np.array(np.hstack((-np.ones_like(nr_border), np.zeros_like(nr_center), np.ones_like(nr_border))))
    else:
        print('Wrong neighborhood passed. Exiting.')
        return None

    lind = np.ravel_multi_index(np.indices(im.shape), im.shape)  # linear indices in array form
    lindv = np.reshape(lind, npts)  # linear indices in vector form
    coordsv = np.array(np.unravel_index(lindv, im.shape))  # coords in array [dim * nvoxels]

    neighbors_m = np.zeros((nghood, npts))
    for i in range(npts):
        s, r, c = tuple(coordsv[:, i])
        # if point doesn't lie in the roi then continue with another one
        if not roi[s, r, c]:
            continue
        for nghb in range(nghood):
            rn = r + nr[nghb]
            cn = c + nc[nghb]
            sn = s + ns[nghb]
            row_ko = rn < 0 or rn > (n_rows - 1)
            col_ko = cn < 0 or cn > (n_cols - 1)
            slice_ko = sn < 0 or sn > (n_slices - 1)
            if row_ko or col_ko or slice_ko or not roi[sn, rn, cn]:
                neighbors_m[nghb, i] = np.NaN
            else:
                indexN = np.ravel_multi_index((sn, rn, cn), im.shape)
                neighbors_m[nghb, i] = indexN

    return neighbors_m


def graycomatrix_3D(data, mask=None, connectivity=1, min_int=0, max_int=255):
    warnings.filterwarnings("error")
    ndims = data.ndim

    if data.max() <= 1:
        data = skiexp.rescale_intensity(data, (0, 1), (0, 255)).astype(np.int)

    if mask is None:
        mask = np.ones_like(data)
    mask = np.where(data < min_int, 0, mask)
    mask = np.where(data > max_int, 0, mask)
    mask_v = mask.flatten()

    if ndims == 2:
        if connectivity == 1:
            nghood = 4
        else:
            nghood = 8
    elif ndims == 3:
        if connectivity == 1:
            nghood = 6
        else:
            nghood = 26
    else:
        raise AttributeError('Unsupported image dimension.')

    nghbm = make_neighborhood_matrix(data, nghood=nghood).T

    # n_rows, n_cols = data.shape
    glcm = np.zeros((256, 256), dtype=np.uint32)
    data_v = data.flatten()
    for p, nghbs in enumerate(nghbm):
        if mask_v[p]:
            for n in nghbs:
                if not np.isnan(n):
                    if mask_v[int(n)]:
                        try:
                            glcm[data_v[p], data_v[int(n)]] += 1
                        except:
                            pass

    return glcm


def initialize_graycom(data_in, slice=None, distances=(1, ), scale=0.5, angles=(0, np.pi / 4, np.pi / 2, 3 * np.pi / 4),
                       symmetric=False, c_t=5, show=False, show_now=True):
    if scale != 1:
        data = resize3D(data_in, scale, sliceId=0)
    else:
        data = data_in.copy()

    # computing gray co-occurence matrix
    print('Computing gray co-occurence matrix ...',)
    if data.ndim == 2:
        gcm = skifea.greycomatrix(data, distances, angles, symmetric=symmetric)
        # summing over distances and directions
        gcm = gcm.sum(axis=3).sum(axis=2)
    else:
        gcm = graycomatrix_3D(data, connectivity=1)
    print('done')

    # thresholding graycomatrix (GCM)
    thresh = c_t * np.mean(gcm)
    gcm_t = gcm > thresh
    gcm_to = skimor.binary_opening(gcm_t, selem=skimor.disk(3))

    try:
        blob, seeds, labs_f = blob_from_gcm(gcm_to, data)
        # print 'first'
    except:
        gcm_to = skimor.binary_closing(gcm_t, selem=skimor.disk(3))
        gcm_to = skimor.binary_opening(gcm_to, selem=skimor.disk(3))
        blob, seeds, labs_f = blob_from_gcm(gcm_to, data, slice)
        # print 'second - different gcm processing'

        # plt.figure()
        # plt.subplot(131), plt.imshow(gcm, 'jet', vmax=10 * gcm.mean())
        # plt.subplot(132), plt.imshow(gcm_t)
        # plt.subplot(133), plt.imshow(gcm_to)
        # plt.show()

    # hole filling - adding (and then removing) a capsule of zeros, otherwise it'd fill holes touching image borders
    blob = fill_holes_watch_borders(blob)

    if scale != 1:
        blob = blob.astype(np.uint8)
        if blob.ndim == 2:
            blob = cv2._resize_if_required(blob, data_in.shape[::-1])
        else:
            tmp = np.zeros(data_in.shape)
            for i, im in enumerate(blob):
                tmp[i, :, :] = cv2._resize_if_required(im, (data_in.shape[2], data_in.shape[1]))
            blob = tmp

    # visualization
    if show:
        if slice is None:
            slice = 0
        data_vis = data if data.ndim == 2 else data[slice, ...]
        seeds_vis = seeds if data.ndim == 2 else seeds[slice, ...]
        labs_f_vis = labs_f if data.ndim == 2 else labs_f[slice, ...]
        blob_vis = blob if data.ndim == 2 else blob[slice, ...]

        plt.figure()
        plt.subplot(131), plt.imshow(gcm, 'gray', vmax=gcm.mean()), plt.title('gcm')
        plt.subplot(132), plt.imshow(gcm_t, 'gray'), plt.title('thresholded')
        plt.subplot(133), plt.imshow(gcm_to, 'gray'), plt.title('opened')

        plt.figure()
        plt.subplot(121), plt.imshow(data_vis, 'gray', interpolation='nearest'), plt.title('input')
        plt.subplot(122), plt.imshow(seeds_vis, 'jet', interpolation='nearest'), plt.title('seeds')
        divider = make_axes_locatable(plt.gca())
        cax = divider.append_axes('right', size='5%', pad=0.05)
        plt.colorbar(cax=cax, ticks=np.unique(seeds_vis))

        plt.figure()
        plt.subplot(131), plt.imshow(data_vis, 'gray'), plt.title('input')
        plt.subplot(132), plt.imshow(labs_f_vis, 'gray'), plt.title('labels')
        plt.subplot(133), plt.imshow(blob_vis, 'gray'), plt.title('init blob')

        if show_now:
            plt.show()

    return blob


def analyze_glcm(glcm, area_t=200, ecc_t=0.35, show=False, show_now=True, verbose=False):
    glcm = skimor.binary_closing(glcm)
    labs_im = skimea.label(glcm, connectivity=2)

    # plt.figure()
    # plt.subplot(121), plt.imshow(glcm, 'gray', interpolation='nearest')
    # plt.subplot(122), plt.imshow(labs_im, 'jet', interpolation='nearest')
    # plt.show()

    blobs = describe_blob(labs_im, area_t=area_t, ecc_t=ecc_t, verbose=verbose)
    means = [np.array(b.centroid).mean() for b in blobs]
    stds = 5 / np.array([b.eccentricity for b in blobs])

    rvs = [scista.norm(m, s) for m, s in zip(means, stds)]

    if show:
        imv = cv2.cvtColor(255 * glcm.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        for b in blobs:
            # prop = skimea.regionprops(b)
            # centroid = tuple(map(int, np.round(prop.centroid)))
            # major_axis = int(round(prop.major_axis_length / 2))
            # minor_axis = int(round(prop.minor_axis_length / 2))
            # angle = int(round(np.degrees(prop.orientation)))
            cv2.ellipse(imv, b.centroid, (b.minor_axis, b.major_axis), b.angle, 0, 360, (255, 0, 255), thickness=2)
        plt.figure()
        plt.imshow(imv)
        plt.axis('off')
        if show_now:
            plt.show()
        #
        # plt.figure()
        # plt.imshow(imv, 'gray', interpolation='nearest')
        # plt.hold(True)
        # plt.plot(centroid[0], centroid[1], 'ro')
        # plt.plot(c1, r1, 'go')
        # plt.plot(c2, r2, 'go')
        # plt.plot(new_cent1[0], new_cent1[1], 'co')
        # plt.plot(new_cent2[0], new_cent2[1], 'yo')
        # plt.axis('image')
        # plt.show()

    return rvs


def blob_from_gcm(gcm, data, slice=None, show=False, show_now=True, return_rvs=False):
    rvs = analyze_glcm(gcm, show=show, show_now=show_now)

    if slice is not None:
        data = data[slice, ...]

    seeds = np.zeros(data.shape, dtype=np.uint8)
    best_probs = np.zeros(data.shape)  # assign the most probable label if more than one are possible
    for i, rv in enumerate(rvs):
        probs = rv.pdf(data)
        s = probs > probs.mean()  # probability threshold
        s = fill_holes_watch_borders(s)
        # s = skimor.binary_opening(s, selem=skimor.disk(3))
        # plt.figure()
        # plt.subplot(131), plt.imshow(s, 'gray')
        # plt.subplot(132), plt.imshow(sfh, 'gray')
        # plt.subplot(133), plt.imshow(sfh2, 'gray')
        # plt.show()
        s = np.where((probs * s) > (best_probs * s), i + 1, s)  # assign new label only if its probability is higher
        best_probs = np.where(s, probs, best_probs)  # update best probs
        seeds = np.where(s, i + 1, seeds)  # update seeds

    labs_f = scindifil.median_filter(seeds, size=3)

    # finding biggest blob - this would be our initialization
    adepts_lbl, n_labels = skimea.label(labs_f, connectivity=2, return_num=True)
    areas = [(adepts_lbl == l).sum() for l in range(1, n_labels + 1)]
    blob = adepts_lbl == (np.argmax(areas) + 1)

    if return_rvs:
        return blob, seeds, labs_f, rvs
    else:
        return blob, seeds, labs_f


def describe_blob(labs_im, area_t=200, ecc_t=0.25, verbose=False):
    # TODO: misto ecc kontrolovat jen major_axis?
    props = skimea.regionprops(labs_im)
    blobs = []
    blob = namedtuple('blob', ['label', 'area', 'centroid', 'eccentricity', 'major_axis', 'minor_axis', 'angle'])
    for i, prop in enumerate(props):
        label = prop.label
        area = int(prop.area)
        centroid = map(int, prop.centroid)
        major_axis = prop.major_axis_length
        minor_axis = prop.minor_axis_length
        angle = prop.orientation
        # my_ecc = minor_axis / major_axis
        try:
            eccentricity = minor_axis / major_axis
        except ZeroDivisionError:
            eccentricity = 0
        msg = '#{}: area={}, centroid={}, eccentricity={:.2f}'.format(i, area, centroid, eccentricity)

        if (area > area_t) and (eccentricity > ecc_t):
            centroid = tuple(map(int, np.round(prop.centroid)))
            major_axis = int(round(prop.major_axis_length / 2))
            minor_axis = int(round(prop.minor_axis_length / 2))
            angle = int(round(np.degrees(prop.orientation)))
            blobs.append(blob(label, area, centroid, eccentricity, major_axis, minor_axis, angle))
            msg += '... OK'
        elif area <= area_t:
            msg += '... TO SMALL - DISCARDING'
        elif eccentricity <= ecc_t:
            msg += '... SPLITTING'
            splitted = split_blob(labs_im == label, prop)
            for spl in splitted:
                spl_blob = describe_blob(spl)
                blobs += spl_blob
                pass
        _debug(msg, verbose)

    return blobs


def split_blob(im, prop):
    # blobs = []
    # blob = namedtuple('blob', ['label', 'area', 'centroid', 'eccentricity'])
    centroid = tuple(map(int, np.round(prop.centroid)))
    major_axis = int(round(prop.major_axis_length / 2))
    minor_axis = int(round(prop.minor_axis_length / 2))
    angle = int(round(np.degrees(prop.orientation)))

    c1 = centroid[1] + major_axis * np.cos(prop.orientation)
    r1 = centroid[0] - major_axis * np.sin(prop.orientation)
    c2 = centroid[1] + major_axis * np.cos(prop.orientation + np.pi)
    r2 = centroid[0] - major_axis * np.sin(prop.orientation + np.pi)

    new_cent1 = ((centroid[0] + r1) / 2, (centroid[1] + c1) / 2)
    new_cent1 = tuple(map(int, map(round, new_cent1)))
    new_cent2 = ((centroid[0] + r2) / 2, (centroid[1] + c2) / 2)
    new_cent2 = tuple(map(int, map(round, new_cent2)))


    # imv = cv2.cvtColor(255 * im.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    # cv2.ellipse(imv, centroid, (minor_axis, major_axis), angle, 0, 360, (0, 0, 255), thickness=2)
    # cv2.ellipse(imv, new_cent1, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, (255, 0, 0), thickness=2)
    # cv2.ellipse(imv, new_cent2, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, (255, 0, 0), thickness=2)
    #
    # plt.figure()
    # plt.imshow(imv, 'gray', interpolation='nearest')
    # plt.hold(True)
    # plt.plot(centroid[0], centroid[1], 'ro')
    # plt.plot(c1, r1, 'go')
    # plt.plot(c2, r2, 'go')
    # plt.plot(new_cent1[0], new_cent1[1], 'co')
    # plt.plot(new_cent2[0], new_cent2[1], 'yo')
    # plt.axis('image')
    # plt.show()

    im1 = np.zeros_like(im).astype(np.uint8)
    cv2.ellipse(im1, new_cent1, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, 1, thickness=-1)
    im1 *= im

    im2 = np.zeros_like(im).astype(np.uint8)
    cv2.ellipse(im2, new_cent2, (minor_axis, int(round(major_axis / 2))), angle, 0, 360, 1, thickness=-1)
    im2 *= im

    # plt.figure()
    # plt.subplot(121), plt.imshow(im1, 'gray')
    # plt.subplot(122), plt.imshow(im2, 'gray')
    # plt.show()
    #
    # blob1 = blob(prop.label, prop.area, new_cent1, 0.5)
    # blob2 = blob(prop.label, prop.area, new_cent2, 0.5)

    # return [blob1, blob2]
    return (im1, im2)


def visualize_seg(data, seg, mask=None, slice=None, title='visualization of segmentation', show_now=True, for_save=False):
    if slice is None:
        slice = 0
    data_vis = data if data.ndim == 2 else data[slice,...]
    seg_vis = seg if data.ndim == 2 else seg[slice,...]
    data_vis = data_vis.astype(np.uint8)
    seg_vis = seg_vis.astype(np.uint8)
    if mask is not None:
        mask_vis = mask if data.ndim == 2 else mask[slice,...]
        mask_vis = mask_vis.astype(np.uint8)

    seg_over = skicol.label2rgb(seg_vis, image=data_vis, colors=['red', 'green', 'blue'], bg_label=0)
    seg_bounds = skiseg.mark_boundaries(data_vis, seg_vis, color=(1, 0, 0), mode='thick')
    # seg_over = seg_vis
    # seg_bounds = seg_vis

    if mask is not None:
        try:
            mask_bounds = skiseg.mark_boundaries(data_vis, mask_vis, color=(1, 0, 0), mode='thick')
        except:
            pass
        # mask_bounds = mask_vis
        if for_save:
            fig = plt.figure(figsize=(20, 10))
        else:
            fig = plt.figure()
        plt.suptitle(title)
        plt.subplot(231), plt.imshow(data_vis, 'gray'), plt.title('input'), plt.axis('off')
        plt.subplot(232), plt.imshow(mask_vis, 'gray'), plt.title('init mask'), plt.axis('off')
        plt.subplot(233), plt.imshow(mask_bounds, 'gray'), plt.title('init mask'), plt.axis('off')
        plt.subplot(234), plt.imshow(seg_vis, 'gray'), plt.title('segmentation'), plt.axis('off')
        plt.subplot(235), plt.imshow(seg_over, 'gray'), plt.title('segmentation'), plt.axis('off')
        plt.subplot(236), plt.imshow(seg_bounds, 'gray'), plt.title('segmentation'), plt.axis('off')
    else:
        if for_save:
            fig = plt.figure(figsize=(20, 5))
        else:
            fig = plt.figure()
        plt.suptitle(title)
        plt.subplot(141), plt.imshow(data_vis, 'gray'), plt.title('input'), plt.axis('off')
        plt.subplot(142), plt.imshow(seg_vis, 'gray'), plt.title('segmentation'), plt.axis('off')
        plt.subplot(143), plt.imshow(seg_over, 'gray'), plt.title('segmentation'), plt.axis('off')
        plt.subplot(144), plt.imshow(seg_bounds, 'gray'), plt.title('segmentation'), plt.axis('off')
    if show_now:
        plt.show()

    return fig


def hist_smoothing(bins, hist, window='gaussian', win_w=20, sigma=5):
    # # generated a density class
    # density = scista.gaussian_kde(hist)
    #
    # # set the covariance_factor, lower means more detail
    # density.covariance_factor = lambda: 0.25
    # density._compute_covariance()
    #
    # # generate a fake range of x values
    # xs = np.arange(0, 24, .1)
    #
    # # fill y values using density class
    # ys = density(xs)
    #
    # plt.figure()
    # plt.fill(bins, hist, 'b', bins, ys, 'r', alpha=0.5)
    # plt.show()
    win_types = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman', 'gaussian']
    if not window in win_types:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    # for window in win_types:
    if window == 'flat':  # moving average
        w = np.ones(win_w, 'd')
    elif window == 'gaussian':
        w = scisig.gaussian(win_w, std=sigma)
    else:
        w = eval('np.' + window + '(win_w)')

    # w = np.hamming(win_w)
    y = np.convolve(w / w.sum(), hist, mode='same')

    #     plt.figure()
    #     plt.fill(bins, hist, 'b', bins, y, 'r', alpha=0.5)
    #     plt.title(window)
    # plt.show()
    return y


def peak_in_hist(bins, hist, min_distance=3):
    inds = skifea.peak_local_max(hist, min_distance=min_distance, indices=True)
    # inds = np.nonzero(mask)

    # plt.figure()
    # plt.fill(bins, hist, 'b', alpha=1)
    # for i in inds:
    #     plt.plot(bins[i], hist[i], 'ro', markersize=15)
    # plt.show()

    return tuple(inds.flatten())


def seeds_from_hist(img, mask=None, window='hanning', smooth=True, min_distance=3, seed_area_width=5, min_int=0,
                    max_int=255, show=False, show_now=True, verbose=False):
    if mask is None:
        mask = np.ones_like(img)

    img *= mask
    # odstraneni bodu s velkym gradientem
    edge = np.zeros(img.shape)
    edge = skifil.scharr(img)
    edge = skiexp.rescale_intensity(em, out_range=(0, 1))
    edge_t = 0.1
    pts = img[np.nonzero(edge < edge_t)]

    # odstraneni bodu s krajnimi intenzitami
    pts = pts[pts >= min_int]
    pts = pts[pts <= max_int]

    # urceni histogramu
    hist, bins = skiexp.histogram(pts)

    # ----------
    # pts2 = img[np.nonzero(mask)]
    # pts2 = pts2[pts2 >= min_int]
    # pts2 = pts2[pts2 <= max_int]
    # hist2, bins2 = skiexp.histogram(pts2)
    # plt.figure()
    # # plt.subplot(211)
    # plt.plot(bins2, hist2, color='r', lw=2)
    # # plt.fill_between(bins2, hist2, color='b', lw=2)
    # plt.plot(bins2, hist_smoothing(bins2, hist2, window=window), color='b', lw=2)
    # # plt.subplot(212)
    # plt.figure()
    # plt.plot(bins, hist, color='r', lw=2)
    # # plt.fill_between(bins, hist, color='b', lw=2)
    # plt.plot(bins, hist_smoothing(bins, hist, window=window), color='b', lw=2)
    # plt.show()
    # ----------

    if smooth:
        hist_o = hist.copy()
        hist = hist_smoothing(bins, hist, window=window)

    # vyhledani peaku v histogramu
    inds = peak_in_hist(bins, hist, min_distance=min_distance)
    peaks = [bins[x] for x in inds]

    # urceni intervalu jednotlivych oblasti
    seeds_intervals = []
    for i in peaks:
        seed_min = max(i - seed_area_width, 0)
        seed_max = min(i + seed_area_width, 255)
        seeds_intervals.append((seed_min, seed_max))
        # TODO: vyresit mozne prekryvy intervalu

    # urceni seedu
    seeds = np.zeros(img.shape)
    for i, interval in enumerate(seeds_intervals):
        tmp = (img >= interval[0]) * (img <= interval[1])
        # tmp = skimor.binary_closing(tmp, selem=skimor.disk(1))
        # tmp = skimor.binary_opening(tmp, selem=skimor.disk(1))
        seeds += (i + 1) * tmp

    # vypisy
    if verbose:
        print('peaks:', peaks)
        print('seed intervals:', seeds_intervals)

    # vizualizace
    if show:
        color_iter = itertools.cycle(['g', 'c', 'm', 'y', 'k'])
        plt.figure()
        if smooth:
            plt.plot(bins, hist_o, 'r', linewidth=2)
        # plt.fill(bins, hist_s, 'b', alpha=1)
        plt.plot(bins, hist, 'b', linewidth=3)
        for i, (ind, interval, color) in enumerate(zip(inds, seeds_intervals, color_iter)):
            plt.plot(bins[ind], hist[ind], color + 'o', markersize=15)
            plt.plot((interval[0], interval[0]), (0, hist[ind] + 10), color + '-', linewidth=3)
            plt.plot((interval[1], interval[1]), (0, hist[ind] + 10), color + '-', linewidth=3)
        plt.xlim(xmin=0, xmax=255)

        plt.figure()
        plt.subplot(131), plt.imshow(img, 'gray'), plt.axis('off')
        plt.subplot(132), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(scindifil.median_filter(seeds, size=3), 'jet', interpolation='nearest'), plt.axis('off')
        # divider = make_axes_locatable(plt.gca())
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # plt.colorbar(cax=cax, ticks=range(len(peaks) + 1))

        if show_now:
            plt.show()
    return seeds, peaks


def seeds_from_glcm(img, mask=None, smooth=True, min_int=0, max_int=255, show=False, show_now=True, verbose=False):
    glcm = graycomatrix_3D(img, mask=mask)

    # thresholding graycomatrix (GCM)
    c_t = 5
    thresh = c_t * np.mean(glcm)
    glcm_t = glcm > thresh
    glcm_to = skimor.binary_closing(glcm_t, selem=skimor.disk(3))
    glcm_to = skimor.binary_opening(glcm_to, selem=skimor.disk(3))
    blob, seeds, labs_f, rvs = blob_from_gcm(glcm_to, img, return_rvs=True, show=show, show_now=False)

    centers = [x.mean() for x in rvs]

    if verbose:
        print('centers:', centers)

    if show:
        plt.figure()
        plt.subplot(131), plt.imshow(img, 'gray'), plt.title('input'), plt.axis('off')
        plt.subplot(132), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.title('seeds'), plt.axis('off')
        plt.subplot(133), plt.imshow(labs_f, 'jet', interpolation='nearest'), plt.title('filtered seeds'), plt.axis('off')
        if show_now:
            plt.show()

    return seeds, centers


def seeds_from_glcm_mesh(img, mask=None, smooth=True, min_int=0, max_int=255, show=False, show_now=True, verbose=False):
    mask = (img > 0) * (img < 255)
    glcm = graycomatrix_3D(img, mask=mask)

    # deriving max number of classes
    _, peaks = seeds_from_hist(img, min_int=5, max_int=254)
    num_peaks = len(peaks)

    # finding local maxima in glcm
    inds = skifea.peak_local_max(np.triu(glcm), min_distance=20, exclude_border=False, indices=True, num_peaks=num_peaks)
    inds = np.array([int(round(np.mean(x))) for x in inds])

    # sorting the maxima
    min_coo = -1
    peaks_str = np.array([glcm[x, x] for x in inds])
    sorted_idxs = np.argsort(peaks_str)[::-1]
    inds = inds[sorted_idxs]

    # finding labels
    glcm_c = skimor.closing(glcm, selem=skimor.disk(3))
    class_vals = [[] for i in range(len(inds))]
    labels = []
    for x in range(256):
        if glcm_c[x, x] > min_coo:
            dists = abs(inds - x)
            idx = np.argmin(dists)
            labels.append(idx)
            class_vals[idx].append(x)
        else:
            labels.append(-1)

    # deriving  class ellipses and transition lines
    ellipses = []
    trans = []
    for i, c in zip(inds, class_vals):
        if c:
            c = np.array(c)
            trans.append(c.max())
            i = int(round((c.max() + c.min()) / 2.))
            cent = [i, i]
            major_axis = ((c.max() - c.min()) + 1) / 0.7071  # / 2
            ellipses.append((cent, major_axis))

    # deriving seeds
    seeds = np.array(labels)[img.flatten()].reshape(img.shape)
    seeds_f = scindifil.median_filter(seeds, size=3)

    if verbose:
        print('inds:', inds)

    # visualization
    if show:
        # seeds
        plt.figure()
        plt.subplot(131), plt.imshow(img, 'gray'), plt.axis('off')
        plt.subplot(132), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(seeds_f, 'jet', interpolation='nearest'), plt.axis('off')

        # plt.figure()
        # glcm_lbls
        # plt.suptitle('nearest neighbor')
        # plt.subplot(121), plt.imshow(glcm, 'jet')
        # plt.subplot(122), plt.imshow(glcm, 'jet')

        # peaks
        plt.figure()
        plt.subplot(121), plt.imshow(glcm, 'jet')
        plt.axis('off')
        plt.subplot(122), plt.imshow(glcm, 'jet')
        plt.hold(True)
        for i in inds:
            plt.plot(i, i, 'wo')# , markersize=14)
        plt.axis('image')
        plt.axis('off')

        # ellipses and transition lines
        plt.figure()
        plt.imshow(glcm)
        ax = plt.gca()
        for e in ellipses:
            ell = Ellipse(xy=e[0], width=e[1], height=15, angle=45, color='m', ec='k', lw=4)
            ax.add_artist(ell)
        for i in trans:
            plt.plot((2 * i + 1, 0), (0, 2 * i + 1), 'k-', lw=4)
        plt.axis('image')
        plt.axis('off')
        plt.axis([0, 255, 255, 0])

        if show_now:
            plt.show()

    return seeds_f, inds


def data_from_glcm(glcm):
    pts = np.argwhere(glcm)
    counts = [glcm[tuple(i)] for i in pts]
    data = []
    for p, c in zip(pts, counts):
        for i in range(c):
            data.append(p)
    data = np.array(data)
    return data


def _debug(msg, verbose=True, newline=True):
    if verbose:
        if newline:
            print(msg)
        else:
            print(msg,)


def seeds_from_glcm_meanshift(img, mask=None, smooth=True, min_int=0, max_int=255, show=False, show_now=True, verbose=False):
    _debug('calculating glcm ...', verbose, False)
    mask = (img > min_int) * (img < max_int)
    glcm = graycomatrix_3D(img, mask=mask)
    _debug('done', verbose)

    _debug('filtering glcm ...', verbose, False)
    min_num = 2 * glcm.mean()
    # plt.figure()
    # plt.subplot(231), plt.imshow(glcm > 0, 'gray')
    # plt.subplot(232), plt.imshow(glcm > (0.1 * glcm.mean()), 'gray')
    # plt.subplot(233), plt.imshow(glcm > (0.5 * glcm.mean()), 'gray')
    # plt.subplot(234), plt.imshow(glcm > (0.8 * glcm.mean()), 'gray')
    # plt.subplot(235), plt.imshow(glcm > glcm.mean(), 'gray')
    # plt.subplot(236), plt.imshow(glcm > (2 * glcm.mean()), 'gray')
    # plt.show()
    glcm = np.where(glcm < min_num, 0, glcm)

    # removing pts that are far from diagonal
    diag = np.ones(glcm.shape)
    k = 20
    tu = np.triu(diag, -k)
    tl = np.tril(diag, k)
    diag = tu * tl
    # plt.figure()
    # plt.subplot(231), plt.imshow(tu, 'gray')
    # plt.subplot(232), plt.imshow(tl, 'gray')
    # plt.subplot(233), plt.imshow(diag, 'gray')
    # plt.subplot(234), plt.imshow(glcm, 'jet', vmax=glcm.mean())
    # plt.subplot(235), plt.imshow(glcm * diag.astype(glcm.dtype), 'jet', vmax=glcm.mean())
    # plt.show()
    glcm *= diag.astype(glcm.dtype)
    # glcm = skimor.closing(glcm, skimor.disk(2))
    _debug('done', verbose)

    _debug('preparing data ...', verbose, False)
    data = data_from_glcm(glcm)
    _debug('done', verbose)

    _debug('estimating bandwidth ...', verbose, False)
    bandwidth = estimate_bandwidth(data, quantile=0.08, n_samples=2000)
    _debug('done', verbose)

    _debug('fitting mean shift ...', verbose, False)
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # ms = MeanShift()
    ms.fit(data)
    labels = ms.labels_
    cluster_centers = ms.cluster_centers_
    _debug('done', verbose)

    labels_unique = np.unique(labels)
    n_clusters_ = len(labels_unique)

    _debug('number of estimated clusters : %d' % n_clusters_, verbose)
    _debug('cluster centers: {}'.format(cluster_centers), verbose)

    # deriving seeds
    int_labels = []
    for x in range(256):
        int_labels.append(ms.predict(np.array([x, x]).reshape(1, -1)))
    seeds = np.array(int_labels)[img.flatten()].reshape(img.shape) + 1
    seeds_f = scindifil.median_filter(seeds, size=3)

    #cluster centers
    centers = [int(round(np.array(x).mean())) for x in ms.cluster_centers_]

    # visualization
    if show:
        plt.figure()
        plt.suptitle('img | seeds | filtered seeds')
        plt.subplot(131), plt.imshow(img, 'gray'), plt.axis('off')
        plt.subplot(132), plt.imshow(seeds, 'jet', interpolation='nearest'), plt.axis('off')
        plt.subplot(133), plt.imshow(seeds_f, 'jet', interpolation='nearest'), plt.axis('off')

        plt.figure()
        plt.subplot(121), plt.imshow(glcm, 'jet')
        for c in cluster_centers:
            plt.plot(c[0], c[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=8)
        plt.axis('image')
        plt.axis('off')
        plt.subplot(122), plt.imshow(glcm, 'jet')
        colors = itertools.cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_), colors):
            my_members = labels == k
            cluster_center = cluster_centers[k]
            plt.plot(data[my_members, 0], data[my_members, 1], col + '.')
            plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor='w', markeredgecolor='k', markersize=8)
        plt.title('Estimated number of clusters: %d' % n_clusters_)
        plt.axis('image')
        plt.axis('off')

        if show_now:
            plt.show()

    return seeds_f, centers


def match_size(d, shape, verbose=False):
    zoom = np.array(shape) / np.array(d.shape).astype(np.float)
    r = scindiint.zoom(d, zoom, order=1, prefilter=False)
    if verbose:
        print('in: {}, req: {}, zoom: {}, out: {}'.format(d.shape, shape, zoom, r.shape))
    return r


def segmentation_accuracy(mask, gt):
    if mask.shape != gt.shape:
        mask = resize_ND(mask, shape=gt.shape)
    precision = (mask * gt).sum() / float(mask.sum())  # how many selected items are relevant
    recall = (mask * gt).sum() / float(gt.sum())  # how many relevant items are selected
    if precision + recall == 0:
        f_measure = 0
    else:
        f_measure = 2 * precision * recall / float((precision + recall))

    # print f_measure, precision, recall
    # plt.figure()
    # plt.subplot(121), plt.imshow(gt, 'gray', interpolation='nearest'), plt.title('gt')
    # plt.subplot(122), plt.imshow(mask, 'gray', interpolation='nearest'), plt.title('seg')
    # plt.show()

    return precision, recall, f_measure

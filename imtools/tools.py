__author__ = 'Ryba'
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as matpat
import os
import sys
import glob

import skimage.exposure as skiexp
import skimage.measure as skimea
import skimage.morphology as skimor
import skimage.transform as skitra
import skimage.filters as skifil
import skimage.restoration as skires
import skimage.segmentation as skiseg
import skimage.feature as skifea
import skimage.io as skiio
from skimage.segmentation import mark_boundaries

import scipy.stats as scista
import scipy.ndimage.morphology as scindimor
import scipy.ndimage.measurements as scindimea
import scipy.ndimage.interpolation as scindiint

import cPickle as pickle
import gzip

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


def windowing(data, level=50, width=350, sub1024=False, sliceId=2):
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
        data = skiexp.rescale_intensity(data, in_range=(minHU, maxHU), out_range=(0, 255))

    return data.astype(np.uint8)


def smoothing(data, d=10, sigmaColor=10, sigmaSpace=10, sliceId=2):
    import cv2
    if data.ndim == 3:
        if sliceId == 2:
            for idx in range(data.shape[2]):
                data[:, :, idx] = cv2.bilateralFilter( data[:, :, idx], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )
        elif sliceId == 0:
            for idx in range(data.shape[0]):
                data[idx, :, :] = cv2.bilateralFilter( data[idx, :, :], d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace )
    else:
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



def analyse_histogram(data, roi=None, show=False, show_now=True, dens_min=20, dens_max=255, minT=0.95, maxT=1.05):
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

    liver = data * (roi > 0)
    class1 = np.where( (liver >= class1TMin) * (liver <= class1TMax), 1, 0)

    if show:
        plt.figure()
        plt.plot(bins, hist)
        plt.hold(True)

        plt.plot(bins[max_peakIdx], hist[max_peakIdx], 'ro')
        plt.plot(bins[histTIdxs], hist[histTIdxs], 'r')
        plt.plot(bins[histTIdxs[0]], hist[histTIdxs[0]], 'rx')
        plt.plot(bins[histTIdxs[-1]], hist[histTIdxs[-1]], 'rx')
        plt.title('Histogram of liver density and its class1 = maximal peak (red dot) +-5% of its density (red line).')
        if show_now:
            plt.show()

    return class1


def intensity_probability(data, std=20, mask=None, dens_min=10, dens_max=255):
    if mask == None:
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

    print('liver pdf: mu = %i, std = %i'%(mu, std))

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


def get_zunics_compatness(obj):
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
    if selem is None:
        if len(data.shape) == 3:
            selem = np.ones((selem_size, selem_size, selem_size))
        else:
            selem = skimor.disk(selem_size)
    if slicewise:
        if sliceId == 0:
            for i in range(data.shape[0]):
                data[i, :, :] = skimor.binary_erosion(data[i, :, :], selem)
        elif sliceId == 2:
            for i in range(data.shape[2]):
                data[:, :, i] = skimor.binary_erosion(data[:, :, i], selem)
    else:
        data = scindimor.binary_erosion(data, selem)
    return data


def resize3D(data, scale, sliceId=2, method='cv2'):
    import cv2
    if sliceId == 2:
        n_slices = data.shape[2]
        # new_shape = cv2.resize(data[:,:,0], None, fx=scale, fy=scale).shape
        new_shape = scindiint.zoom(data[:,:,0], scale).shape
        new_data = np.zeros(np.hstack((new_shape,n_slices)), dtype=np.int)
        for i in range(n_slices):
            # new_data[:,:,i] = cv2.resize(data[:,:,i], None, fx=scale, fy=scale)
            # new_data[:,:,i] = (255 * skitra.rescale(data[:,:,0], scale)).astype(np.int)
            if method == 'cv2':
                new_data[:,:,i] = cv2.resize(data[:,:,i], (0,0),  fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                new_data[:,:,i] = scindiint.zoom(data[:,:,i], scale)
    elif sliceId == 0:
        n_slices = data.shape[0]
        # new_shape = cv2.resize(data[0,:,:], None, fx=scale, fy=scale).shape
        # new_shape = skitra.rescale(data[0,:,:], scale).shape
        new_shape =  scindiint.zoom(data[0,:,:], scale).shape
        new_data = np.zeros(np.hstack((n_slices, new_shape)), dtype=np.int)
        for i in range(n_slices):
            # new_data[i,:,:] = cv2.resize(data[i,:,:], None, fx=scale, fy=scale)
            # new_data[i,:,:] = (255 * skitra.rescale(data[i,:,:], scale)).astype(np.int)
            if method == 'cv2':
                new_data[i,:,:] = cv2.resize(data[i,:,:], (0,0),  fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            else:
                new_data[i,:,:] = scindiint.zoom(data[i,:,:], scale)
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
        print 'Unknown color, using red as default.'
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
        print 'labels: ', np.unique(labs)
        # py3DSeedEditor.py3DSeedEditor(labs).show()
        for l in range(n_labels + 1):
            labels = np.where(labs == l, curr_l, labels)
            curr_l += 1
    print 'min = %i, max = %i' % (labels.min(), labels.max())
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
        print get_status_text('\tSaving figures', iter=s, max_iter=len(imgs)),
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
    print get_status_text('\tSaving figures', iter=-1, max_iter=len(imgs))


def view_segmentation(datap_1, datap_2=None):
    if not os.path.exists('../data_viewers/'):
        print 'Package data_viewers not found.'
    else:
        sys.path.append('../data_viewers/')
        from dataviewers.seg_viewer import SegViewer

        from PyQt4 import QtGui
        app = QtGui.QApplication(sys.argv)
        le = SegViewer(datap1=datap_1, datap2=datap_2)
        le.show()
        sys.exit(app.exec_())


def show_3d(data, range=True):
    if not os.path.exists('../data_viewers/'):
        print 'Package data_viewers not found.'
    else:
        sys.path.append('../data_viewers/')
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

        from PyQt4 import QtGui
        app = QtGui.QApplication(sys.argv)
        viewer = Viewer_3D(data_vis, range=True)
        viewer.show()
        sys.exit(app.exec_())


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
    resized = cv2.resize(image, dim, interpolation=inter)

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
    import cv2
    if inter is None:
        inter=cv2.INTER_AREA

    w = int(image.shape[1] / scale)
    img = resize(image, width=w, inter=inter)
    if smooth:
        # img = smoothing(image.astype(np.uint8), d=20, sigmaColor=20, sigmaSpace=20).astype(image.dtype)
        img = smoothing(image.astype(np.uint8)).astype(image.dtype)

    if img.shape[0] < min_size[1] or img.shape[1] < min_size[0]:
        return None
    else:
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
    if image.ndim == 2:
        image = np.expand_dims(image, 0)
        window_size = (1, window_size[0], window_size[1])
        if mask is not None:
            mask = np.expand_dims(mask, 0)
    if mask is None:
        mask = np.ones(image.shape, dtype=np.bool)
    # slide a window across the image
    for z in xrange(0, image.shape[0], step_size):
        # c_z = z + window_size[0] / 2.
        for y in xrange(0, image.shape[1], step_size):
            # c_y = y + window_size[2] / 2.
            for x in xrange(0, image.shape[2], step_size):
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


def sigmoid(image, mask=None, a=0.1, c=20, sigm_t = 0.2):
    if mask is None:
        mask = np.ones_like(image)

    im_sigm = (1. / (1 + (np.exp(-a * (image - c))))) * mask
    im_sigm *= (im_sigm > sigm_t)

    return im_sigm
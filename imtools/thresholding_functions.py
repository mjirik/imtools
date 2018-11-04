# -*- coding: utf-8 -*-
"""
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky
Email:       volkovinsky.pavel@gmail.com

Created:     2014/02/22
Copyright:   (c) Pavel Volkovinsky
"""

import logging
logger = logging.getLogger(__name__)

import numpy
import scipy
import scipy.ndimage
from scipy import stats

# import matplotlib
import matplotlib.pyplot as matpyplot



def prepareVisualization(data):

    img0 = numpy.sum(data, axis=0)
    img0[img0 > 0] += numpy.max(img0)

    img1 = numpy.sum(data, axis=1)
    img1[img1 > 0] += numpy.max(img1)

    img2 = numpy.sum(data, axis=2)
    img2[img2 > 0] += numpy.max(img2)

    return img0, img1, img2


def fillHoles(data):
    data_new = scipy.ndimage.binary_fill_holes(
        data).astype(int)
    return data_new


def gaussFilter(data, sigma):
    """

    Aplikace gaussova filtru.

    """

    # Filtrovani
    # print("Gaussfilter 1 ", np.max(data), sigma, np.mean(data))
    scipy.ndimage.filters.gaussian_filter(
        data, sigma, order=0, output=data, mode='nearest')

    # print("Gaussfilter 2 ", np.max(data), np.mean(data))

    return data


def thresholding(data, min_threshold, max_threshold, use_min_threshold=True,
                 use_max_threshold=True):
    """

    Prahovani podle minimalniho a maximalniho prahu.

    """
    tmp = numpy.ones(data.shape, dtype=numpy.bool)
    if use_min_threshold:

        tmp =(data >= min_threshold)
        # out += data >= min_threshold
    if use_max_threshold:
         tmp = tmp & (data <= max_threshold)
        # out += data <= max_threshold

    return tmp
    # return out.astype(numpy.int)


def binaryClosingOpening(data, closeNum, openNum, firstClosing=True,
                         fixBorder=True):
    """

    Aplikace binarniho uzavreni a pote binarniho otevreni.

    """

    # This creates empty border around data, so closing operations wont cut
    # off parts of segmented data on the sides
    data = (data != 0)
    if fixBorder and closeNum >= 1:
        shape = data.shape
        new_shape = (shape[0] + closeNum * 2,
                     shape[1] + closeNum * 2,
                     shape[2] + closeNum * 2)
        logger.debug('Creating empty border for closeing operation...')
        logger.debug('orig shape: ' + str(
            shape) + ' new shape: ' + str(new_shape))

        new_data = numpy.zeros(new_shape, dtype=type(data[0][0][0]))
        new_data[closeNum:closeNum + shape[0],
                 closeNum:closeNum + shape[1],
                 closeNum:closeNum + shape[2]] = data
        data = new_data
        del(new_data)

    # @TODO - not used!!!, why was this here?
    # numpyDataOnes = numpy.ones(data.shape, dtype = type(data[0][0][0]))

    if firstClosing:

        # Vlastni binarni uzavreni.
        if (closeNum >= 1):

            # data = data * \
            #     scipy.ndimage.binary_closing(data, iterations=closeNum)
            data = scipy.ndimage.binary_closing(data, iterations=closeNum)
            logger.debug('closing')

        # Vlastni binarni otevreni.

        if (openNum >= 1):

            data = scipy.ndimage.binary_opening(data, iterations=openNum)
            logger.debug('opening')

    else:

        # Vlastni binarni otevreni.
        if (openNum >= 1):

            data = data * \
                scipy.ndimage.binary_opening(data, iterations=openNum)
            logger.debug('opening')

        # Vlastni binarni uzavreni.
        if (closeNum >= 1):

            data = data * \
                scipy.ndimage.binary_closing(data, iterations=closeNum)
            logger.debug('closing')

    # Removes added empty border. Returns data matrix to original size.
    if fixBorder and closeNum >= 1:
        data = data[closeNum:closeNum + shape[0],
                    closeNum:closeNum + shape[1],
                    closeNum:closeNum + shape[2]]

    return data


def calculateSigma(voxel, input):
    """

    Spocita novou hodnotu sigma pro gaussovo filtr.

    """
    import numpy as np
    voxel = np.asarray(voxel)
    sigma = input / voxel
    return sigma

    # voxelV = voxel[0] * voxel[1] * voxel[2]
    #
    # if (voxel[0] == voxel[1] == voxel[2]):
    #
    #     return ((5 / voxel[0]) * input) / voxelV
    #
    # else:
    #
    #     sigmaX = (5.0 / voxel[0]) * input
    #     sigmaY = (5.0 / voxel[1]) * input
    #     sigmaZ = (5.0 / voxel[2]) * input
    #
    #     return numpy.asarray([sigmaX, sigmaY, sigmaZ]) / voxelV


def calculateAutomaticThresholdOtsu(data, arrSeed=None):
    from skimage.filters import threshold_otsu
    return threshold_otsu(data.reshape(-1))

def calculateAutomaticThreshold(data, arrSeed=None):
    """

    Automaticky vypocet prahu - pokud jsou data bez oznacenych objektu, tak
    vraci nekolik nejvetsich objektu.  Pokud jsou ale definovany prioritni
    seedy, tak je na jejich zaklade vypocitan prah.

    """

    if arrSeed != None:

        threshold = numpy.round(min(arrSeed), 2) - 1
        logger.debug(
            'Zjisten automaticky threshold ze seedu (o 1 zmenseny): ' +
            str(threshold))
        return threshold

    # Hustota hist
    hist_points = 1300

    # Pocet bodu v primce 1 ( klesajici od maxima )
    pointsFrom = 20  # (int)(hist_points * 0.05)

    # Pocet bodu v primce 2 ( stoupajici k okoli spravneho thresholdu)
    pointsTo = 20  # (int)(hist_points * 0.1)

    # Pocet bodu k preskoceni od konce hist
    pointsSkip = (int)(hist_points * 0.025)

    # hledani maxima: zacina se od 'start'*10 procent delky histu (aby se
    # preskocili prvni oscilace)
    start = 0.1

    # hist: funkce(threshold)
    hist, bin_edges = numpy.histogram(data, bins=hist_points)
    # bin_centers: threshold
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # last je maximum z hist
    # init_index je index "last"
    init_index = 0
    last = hist[(int)(len(hist) * start)]
    for index in range((int)(len(hist) * start), len(hist)):
        if(last < hist[index]):
            last = hist[index]  # maximum histu
            init_index = index  # pozice maxima histu

    # muj_histogram_temp == { f(x+1) = hist[x+1] + hist[x] }
    # stoupajici tendence histogramu
    muj_histogram_temp = []
    muj_histogram_temp.insert(0, hist[0])
    for index in range(1, len(hist)):
        muj_histogram_temp.insert(
            index, hist[index] + muj_histogram_temp[index - 1])

    # reverse muj_histogram_temp do muj_histogram
    # klesajici tendence histogramu
    muj_histogram = muj_histogram_temp[::-1]

    """
        1. primka (od maxima)
        """

    # Pridani bodu to poli x1 a y1
    # (klesajici tendence)
    x1 = []
    y1 = []
    place = 0
    for index in range(init_index, init_index + pointsFrom):
        x1.insert(place, bin_centers[index])
        y1.insert(place, muj_histogram[index])
# print("[ " + str(x1[place]) + ", " + str(y1[place]) + " ]")
        place += 1

    # Linearni regrese nad x1 a y1
    # slope == smernice
    # intercept == posuv
    slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1, y1)

    """
        2. primka (k thresholdu)
        """

    x2 = []
    y2 = []
    place = 0
    for index in range(init_index + pointsFrom + pointsSkip,
                       init_index + pointsFrom + pointsSkip + pointsTo):
        x2.insert(place, bin_centers[index])
        y2.insert(place, muj_histogram[index])
        place += 1

    slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(x2, y2)

    threshold = (intercept2 - intercept1) / (slope1 - slope2)
    threshold = numpy.round(threshold, 2)

    logger.info('Threshold: ' + str(threshold))

    return threshold


def histogram(data, interactivity, histogram_points=1000, start=-1, end=-1,
              line=-1):

    # hist: funkce(threshold)
    hist, bin_edges = numpy.histogram(
        data, bins=histogram_points, range=(start, end))
    # bin_centers: threshold
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if interactivity:

        matpyplot.figure(figsize=(11, 4))
        matpyplot.plot(bin_centers, hist, lw=2)
        # matpyplot.plot([1100*slope1],[1200*slope1],label='one', color='green')
        # matpyplot.plot([1100*slope2], [1200*slope2], label='two',
        # color='blue')

        if line != -1:

            matpyplot.axvline(line, color='purple', ls='--', lw=2)

        matpyplot.show()

    # Return values:
    # centers = x
    # histogram = f(x) = f(centers)
    return bin_centers, hist


def get_intensities_on_seed_position(data, seeds):

    from . import image_manipulation as imma
    seeds_inds = imma.as_seeds_inds(seeds, data.shape)
    # import sed3
    # ed = sed3.sed3(data, seeds)
    # ed.show()
    # Zalozeni pole pro ulozeni seedu
    arrSeed = []
    # Zjisteni poctu seedu.
    stop = seeds_inds[0].size
    tmpSeed = 0
    dim = numpy.ndim(data)

    for index in range(0, stop):
        # Tady se ukladaji labely na mistech, ve kterych kliknul uzivatel.
        if dim == 3:
            # 3D data.
            tmpSeed = data[seeds_inds[0][index], seeds_inds[1][index], seeds_inds[2][index]]
        elif dim == 2:
            # 2D data.
            tmpSeed = data[seeds_inds[0][index], seeds_inds[1][index]]

        # Tady opet pocitam s tim, ze oznaceni nulou pripada cerne
        # oblasti (pozadi).
        if not (tmpSeed == 0).all():
            # Pokud se nejedna o pozadi (cernou oblast), tak se
            # novy seed ulozi do pole "arrSeed"
            arrSeed.append(tmpSeed)

    return arrSeed



def getPriorityObjects(*args, **kwargs):
    logger.warning("Function getPriorityObjects has been renamed. Use get_priority_objects().")
    DeprecationWarning("Function getPriorityObjects has been renamed. Use get_priority_objects().")
    return get_priority_objects(*args, **kwargs)


def get_priority_objects(data, nObj=1, seeds=None, seeds_multi_index=None, debug=False):
    """
    Get N biggest objects from the selection or the object with seed.
    Similar function is in image_manipulation.select_objects_by_seeds(). Use it if possible.

    :param data:  labeled ndarray
    :param nObj:  number of objects
    :param seeds: ndarray. Objects on non zero positions are returned
    :param debug: bool.
    :return: binar image with selected objects
    """

    # Oznaceni dat.
    # labels - oznacena data.
    # length - pocet rozdilnych oznaceni.
    if seeds is not None:
        # logger.warning("'seeds' parameter is obsolete. Use 'seeds_multi_index' instead of it.")
        if numpy.array_equal(data.shape, numpy.asarray(seeds).shape):
            seeds_multi_index = numpy.nonzero(seeds)
        else:
            if seeds_multi_index is None:
                logger.debug("Seeds looks to be seeds_multi_index.")
                seeds_multi_index = seeds

    dataLabels, length = scipy.ndimage.label(data)

    logger.info('Olabelovano oblasti: ' + str(length))

    logger.debug('data labels: ' + str(dataLabels))

    # Uzivatel si nevybral specificke objekty.
    if (seeds_multi_index is None):

        logger.info('Vraceni bez seedu')
        logger.debug('Objekty: ' + str(nObj))

        # Zjisteni nejvetsich objektu.
        arrayLabelsSum, arrayLabels = areaIndexes(dataLabels, length)
        # Serazeni labelu podle velikosti oznacenych dat (prvku / ploch).
        arrayLabelsSum, arrayLabels = selectSort(arrayLabelsSum, arrayLabels)

        returning = None
        label = 0
        stop = nObj - 1

        # Budeme postupne prochazet arrayLabels a postupne pridavat jednu
        # oblast za druhou (od te nejvetsi - mimo nuloveho pozadi) dokud
        # nebudeme mit dany pocet objektu (nObj).
        while label <= stop:

            if label >= len(arrayLabels):
                break

            if arrayLabels[label] != 0:
                if returning is None:
                    # "Prvni" iterace
                    returning = data * (dataLabels == arrayLabels[label])
                else:
                    # Jakakoli dalsi iterace
                    returning = returning + data * \
                        (dataLabels == arrayLabels[label])
            else:
                # Musime prodlouzit hledany interval, protoze jsme narazili na
                # nulove pozadi.
                stop = stop + 1

            label = label + 1

            if debug:
                logger.debug(str(label - 1) + ': ' + str(returning))

        if returning is None:
            logger.info(
                'Zadna validni olabelovana data! (DEBUG: returning == None)')

        return returning

    # Uzivatel si vybral specificke objekty (seeds != None).
    else:

        logger.info('Vraceni se seedy')

        # Zalozeni pole pro ulozeni seedu
        arrSeed = []
        # Zjisteni poctu seedu.
        stop = seeds_multi_index[0].size
        tmpSeed = 0
        dim = numpy.ndim(dataLabels)
        for index in range(0, stop):
            # Tady se ukladaji labely na mistech, ve kterych kliknul uzivatel.
            if dim == 3:
                # 3D data.
                tmpSeed = dataLabels[
                    seeds_multi_index[0][index], seeds_multi_index[1][index], seeds_multi_index[2][index]]
            elif dim == 2:
                # 2D data.
                tmpSeed = dataLabels[seeds_multi_index[0][index], seeds_multi_index[1][index]]

            # Tady opet pocitam s tim, ze oznaceni nulou pripada cerne oblasti
            # (pozadi).
            if tmpSeed != 0:
                # Pokud se nejedna o pozadi (cernou oblast), tak se novy seed
                # ulozi do pole "arrSeed"
                arrSeed.append(tmpSeed)

        # Pokud existuji vhodne labely, vytvori se nova data k vraceni.
        # Pokud ne, vrati se "None" typ. { Deprecated: Pokud ne, vrati se cela
        # nafiltrovana data, ktera do funkce prisla (nedojde k vraceni
        # specifickych objektu). }
        if len(arrSeed) > 0:

            # Zbaveni se duplikatu.
            arrSeed = list(set(arrSeed))
            if debug:
                logger.debug('seed list:' + str(arrSeed))

            logger.info(
                'Ruznych prioritnich objektu k vraceni: ' +
                str(len(arrSeed))
            )

            # Vytvoreni vystupu - postupne pricitani dat prislunych specif.
            # labelu.
            returning = None
            for index in range(0, len(arrSeed)):

                if returning is None:
                    returning = data * (dataLabels == arrSeed[index])
                else:
                    returning = returning + data * \
                        (dataLabels == arrSeed[index])

                if debug:
                    logger.debug((str(index)) + ':' + str(returning))

            return returning

        else:

            logger.warning(
                'Zadna validni data k vraceni - zadne prioritni objekty ' +
                'nenalezeny (DEBUG: function getPriorityObjects:' +
                str(len(arrSeed) == 0))
            return None


def areaIndexes(labels, num):
    """

    Zjisti cetnosti jednotlivych oznacenych ploch (labeled areas)
        input:
            labels - data s aplikovanymi oznacenimi
            num - pocet pouzitych oznaceni

        returns:
            dve pole - prvni sumy, druhe indexy

    """

    arrayLabelsSum = []
    arrayLabels = []
    for index in range(0, num + 1):
        arrayLabels.append(index)
        sumOfLabel = numpy.sum(labels == index)
        arrayLabelsSum.append(sumOfLabel)

    return arrayLabelsSum, arrayLabels


def selectSort(list1, list2):
    """
    Razeni 2 poli najednou (list) pomoci metody select sort
        input:
            list1 - prvni pole (hlavni pole pro razeni)
            list2 - druhe pole (vedlejsi pole) (kopirujici pozice pro razeni
                podle hlavniho pole list1)

        returns:
            dve serazena pole - hodnoty se ridi podle prvniho pole, druhe
                "kopiruje" razeni
    """

    length = len(list1)
    for index in range(0, length):
        min = index
        for index2 in range(index + 1, length):
            if list1[index2] > list1[min]:
                min = index2
        # Prohozeni hodnot hlavniho pole
        list1[index], list1[min] = list1[min], list1[index]
        # Prohozeni hodnot vedlejsiho pole
        list2[index], list2[min] = list2[min], list2[index]

    return list1, list2

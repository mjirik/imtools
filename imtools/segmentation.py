# -*- coding: utf-8 -*-
"""
    Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

    Author:      Pavel Volkovinsky
    Email:       volkovinsky.pavel@gmail.com

    Created:     2012/11/08
    Copyright:   (c) Pavel Volkovinsky

"""

import sys
sys.path.append("../src/")
sys.path.append("../extern/")
from imtools import uiThreshold
# import uiThreshold
# import thresholding_functions
import logging
logger = logging.getLogger(__name__)
import numpy
import numpy as np
import scipy
import scipy.ndimage

from . import image_manipulation


def vesselSegmentation(
        data, segmentation=-1, threshold=None, voxelsize_mm=[1, 1, 1], inputSigma=-1,
        aoi_dilation_iterations=0, aoi_dilation_structure=None, nObj=10, biggestObjects=False,
        useSeedsOfCompactObjects=False, seeds=None, interactivity=True, binaryClosingIterations=2,
        binaryOpeningIterations=0, smartInitBinaryOperations=False, returnThreshold=False,
        binaryOutput=True, returnUsedData=False, qapp=None, auto_method='', aoi_label=1,
        forbidden_label=None, slab=None, old_gui=False, debug=False):
    """

    Vessel segmentation z jater.

    Input:
        :param data: - CT (nebo MRI) 3D data
        :param segmentation: - zakladni oblast pro segmentaci, oznacena struktura se
        :param stejnymi: rozmery jako "data",
            kde je oznaceni (label) jako:
                1 jatra,
                -1 zajimava tkan (kosti, ...)
                0 jinde
        :param threshold: - prah
        :param voxelsize_mm: - (vektor o hodnote 3) rozmery jednoho voxelu
        :param inputSigma: - pocatecni hodnota pro prahovani
        :param aoi_dilation_iterations: - pocet operaci dilation nad zakladni oblasti pro
            segmentaci ("segmantation")
        :param aoi_dilation_structure: - struktura pro operaci dilation
        :param nObj: - oznacuje, kolik nejvetsich objektu se ma vyhledat - pokud je
            rovno 0 (nule), vraci cela data
        :param biggestObjects: - moznost, zda se maji vracet nejvetsi objekty nebo ne
        :param seeds: - moznost zadat pocatecni body segmentace na vstupu. Je to matice
            o rozmerech jako data. Vsude nuly, tam kde je oznaceni jsou jednicky
               It can be same shape like data, or it can be
               indexes e.g. from np.nonzero(seeds)
        :param interactivity: - nastavi, zda ma nebo nema byt pouzit interaktivni mod
            upravy dat
        :param binaryClosingIterations: - vstupni binary closing operations
        :param binaryOpeningIterations: - vstupni binary opening operations
        :param smartInitBinaryOperations: - logicka hodnota pro smart volbu pocatecnich
            hodnot binarnich operaci (bin. uzavreni a bin. otevreni)
        :param returnThreshold: - jako druhy parametr funkce vrati posledni hodnotu
            prahu
        :param binaryOutput: - zda ma byt vystup vracen binarne nebo ne (binarnim
            vystupem se rozumi: cokoliv jineho nez hodnota 0 je hodnota 1)
        :param returnUsedData: - vrati pouzita data
        :param aoi_label: label of organ where is the target vessel
        :param forbidden_label: int or list of ints. Labels of areas which are not used for segmentation.

    Output:
        filtrovana data

    """
    # self.qapp = qapp

    dim = numpy.ndim(data)
    logger.debug('Dimenze vstupnich dat: ' + str(dim))
    if (dim < 2) or (dim > 3):
        logger.debug('Nepodporovana dimenze dat!')
        logger.debug('Ukonceni funkce!')
        return None

    if seeds is None:
        logger.debug('Funkce spustena bez prioritnich objektu!')

    if biggestObjects:
        logger.debug(
            'Funkce spustena s vracenim nejvetsich objektu => nebude mozne\
vybrat prioritni objekty!')

    if (nObj < 1):
        nObj = 1

    if biggestObjects:
        logger.debug('Vybrano objektu k vraceni: ' + str(nObj))

    logger.debug('Pripravuji data...')

    voxel = numpy.array(voxelsize_mm)

    # Kalkulace objemove jednotky (voxel) (V = a*b*c).
    voxel1 = voxel[0]  # [0]
    voxel2 = voxel[1]  # [0]
    voxel3 = voxel[2]  # [0]
    voxelV = voxel1 * voxel2 * voxel3

    # number je zaokrohleny 2x nasobek objemove jednotky na 2 desetinna mista.
    # number stanovi doporucenou horni hranici parametru gauss. filtru.
    number = (numpy.round((2 * voxelV ** (1.0 / 3.0)), 2))

    if aoi_label is None:
        target_organ_segmentation = np.ones(segmentation.shape)
    else:
        target_organ_segmentation = image_manipulation.select_labels(segmentation, aoi_label, slab)
    # Operace dilatace (dilation) nad samotnymi jatry ("segmentation").
    if(aoi_dilation_iterations > 0.0):
        target_organ_segmentation = scipy.ndimage.binary_dilation(
            input=target_organ_segmentation, structure=aoi_dilation_structure,
            iterations=aoi_dilation_iterations)

    # remove forbidden areas from segmentation
    if forbidden_label is not None:
        forbidden_organ_segmentation = image_manipulation.select_labels(
            segmentation, forbidden_label, slab)
        target_organ_segmentation[forbidden_organ_segmentation] = 0
        del(forbidden_organ_segmentation)

    # Ziskani datove oblasti jater (bud pouze jater nebo i jejich okoli -
    # zalezi, jakym zpusobem bylo nalozeno s operaci dilatace dat).

    preparedData = (data * (target_organ_segmentation))  # .astype(numpy.float)
    logger.debug('Typ vstupnich dat: ' + str(preparedData.dtype))

#    if preparedData.dtype != numpy.uint8:
#        print 'Data nejsou typu numpy.uint8 => muze dojit k errorum'

    if not numpy.can_cast(preparedData.dtype, numpy.float):
        logger.debug(
            'ERROR: (debug message) Data nejsou takoveho typu, aby se daly \
prevest na typ "numpy.float" => muze dojit k errorum')
        logger.debug('Ukoncuji funkci!')
        raise ValueError("Cannot cast input data to numpy.float")
        return None

    if (preparedData == False).all():
        logger.debug(
            'ERROR: (debug message) Jsou spatna data nebo segmentacni matice: \
all is true == data is all false == bad segmentation matrix (if data matrix is \
ok)')
        logger.debug('Ukoncuji funkci!')
        raise ValueError("Wrong input data. All is true == data is all false == bad segmentation matrix (if data matrix is ok)")
        return None

    # del(data)
    # del(segmentation)

    # Nastaveni rozmazani a prahovani dat.
    if(inputSigma == -1):
        inputSigma = number
    if(inputSigma > number):
        inputSigma = number

    # seeds = None
    if biggestObjects == False and\
            seeds is None and interactivity == True and threshold is None:
        if old_gui:

            logger.debug(
                ('Nyni si levym nebo pravym tlacitkem mysi (klepnutim nebo tazenim)\
     oznacte specificke oblasti k vraceni.'))


            import sed3
            pyed = sed3.sed3qt(preparedData, contour=segmentation, windowW=400, windowC=50)
            # pyed.show()
            pyed.exec_()

            # from PyQt4.QtCore import pyqtRemoveInputHook
            # pyqtRemoveInputHook()
            # import ipdb; ipdb.set_trace() #  noqa BREAKPOINT


            seeds = pyed.seeds

            # Zkontrolovat, jestli uzivatel neco vybral - nejaky item musi byt
            # ruzny od nuly.
            if (seeds != 0).any() == False:

                seeds = None
                logger.debug('Zadne seedy nezvoleny => nejsou prioritni objekty.')

            else:

                # seeds * (seeds != 0) ## seeds je n-tice poli indexu nenulovych
                # prvku => item krychle je == krychle[ seeds[0][x], seeds[1][x],
                # seeds[2][x] ]
                seeds = seeds.nonzero()
                logger.debug('Seedu bez nul: ' + str(len(seeds[0])))

    closing = binaryClosingIterations
    opening = binaryOpeningIterations

    if (smartInitBinaryOperations and interactivity):

        if (seeds == None):  # noqa

            closing = 5
            opening = 1

        else:

            closing = 2
            opening = 0

    # Samotne filtrovani
    if interactivity:
        if old_gui:
            uiT = uiThreshold.uiThresholdQt(
                preparedData, voxel=voxel, threshold=threshold,
                interactivity=interactivity, number=number, inputSigma=inputSigma,
                nObj=nObj, biggestObjects=biggestObjects,
                useSeedsOfCompactObjects=useSeedsOfCompactObjects,
                binaryClosingIterations=closing, binaryOpeningIterations=opening,
                seeds=seeds,
                threshold_auto_method=auto_method,
            )

            output = uiT.run()
        else:
            # TODO use all parameters
            import seededitorqt

            se = seededitorqt.QTSeedEditor(preparedData, voxelSize=voxel)
            import imtools.threshold_qsed_plugin
            wg0 = imtools.threshold_qsed_plugin.QtSEdThresholdPlugin(nObj=nObj, debug=debug)
            se.addPlugin(wg0)
            se.exec_()
            output = se.getContours()



    else:
        uiT = uiThreshold.uiThreshold(
            preparedData, voxel=voxel, threshold=threshold,
            interactivity=interactivity, number=number, inputSigma=inputSigma,
            nObj=nObj, biggestObjects=biggestObjects,
            useSeedsOfCompactObjects=useSeedsOfCompactObjects,
            binaryClosingIterations=closing, binaryOpeningIterations=opening,
            seeds=seeds,
            threshold_auto_method=auto_method,
        )
        output = uiT.run()

    # Vypocet binarni matice.
    if output is None:  # noqa

        logger.debug('Zadna data k vraceni! (output == None)')

    elif binaryOutput:

        output[output != 0] = 1

    # Vraceni matice.
    if returnThreshold:

        if returnUsedData:

            return preparedData, output, uiT.returnLastThreshold()

        else:

            return output, uiT.returnLastThreshold()

    else:

        if returnUsedData:

            return preparedData, output

        else:

            return output

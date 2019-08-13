# -*- coding: utf-8 -*-
"""
Purpose:     (CZE-ZCU-FAV-KKY) Liver medical project

Author:      Pavel Volkovinsky, Miroslav Jirik
Email:       volkovinsky.pavel@gmail.com

Created:     2012/11/08
Copyright:   (c) Pavel Volkovinsky
"""

import sys



sys.path.append("../src/")
sys.path.append("../extern/")

import logging as logger



# import scipy.ndimage


import matplotlib
import matplotlib.pyplot as matpyplot
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button  # , RadioButtons

import gc as garbage



from PyQt5 import QtGui, QtWidgets
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar

import numpy as np

from . import thresholding_functions




class uiThresholdQt(QtWidgets.QDialog):
    def __init__(self, *pars, **params):
    # def __init__(self,parent=None):
        parent = None


        QtWidgets.QDialog.__init__(self, parent)
        # super(Window, self).__init__(parent)
        # self.setupUi(self)
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

# set the layout
        layout = QtWidgets.QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        # layout.addWidget(self.button)
        self.setLayout(layout)

    # def set_params(self, *pars, **params):
        # import sed3.sed3

        params["figure"] = self.figure 
        self.uit = uiThreshold(*pars, **params)
        self.uit.on_close_fcn = self.callback_close
        self.uit.on_show_fcn = self.exec_
        # ed.show()
        self.output = None

    def run(self):
        # pass
        return self.uit.run()

    def callback_close(self, uit):
        # self.output = uit

        self.close()

    def get_values(self):
        pass

class uiThreshold:

    """

    UI pro prahovani 3D dat.

    """

    def __init__(self, data, voxel, threshold=None, interactivity=True,
                 number=100.0, inputSigma=-1, nObj=10, biggestObjects=True,
                 useSeedsOfCompactObjects=True,
                 binaryClosingIterations=2, binaryOpeningIterations=0,
                 seeds=None, cmap=matplotlib.cm.Greys_r, fillHoles=True,
                 figure=None, threshold_auto_method='', threshold_upper=None,
                 debug=True
                 ):
        """

        Inicialitacni metoda.
        Input:
            :param data: data pro prahovani, se kterymi se pracuje
            :param voxel: velikost voxelu
            :param threshold:
            :param interactivity: zapnuti / vypnuti gui
            :param number: maximalni hodnota slideru pro gauss. filtrovani (max sigma)
            :param inputSigma: pocatecni hodnota pro gauss. filtr
            :param nObj: pocet nejvetsich objektu k vraceni
            :param biggestObjects: oznacuje, zda se maji vracet nejvetsi objekty
            :param binaryClosingIterations: iterace binary closing
            :param binaryOpeningIterations: iterace binary opening
            :param seeds: matice s kliknutim uzivatele- pokud se maji vracet
                   specifikce objekty. It can be same shape like data, or it can be
                   indexes e.g. from np.nonzero(seeds)
            :param cmap: grey
            :param threshold_auto_method: 'otsu' use otsu threshold, other string use our liver automatic

        """

        logger.debug('Spoustim prahovani dat...')
        self.on_close_fcn = None

        self.errorsOccured = False
        self.inputDimension = np.ndim(data)

        if(self.inputDimension != 3):
            logger.error(
                'Vstup nema 3 dimenze! Ma jich ' + str(self.inputDimension)
                + '.')
            self.errorsOccured = True
            return

        self.on_show_fcn = plt.show
        self.interactivity = interactivity
        self.cmap = cmap
        self.number = number
        self.inputSigma = inputSigma
        # if shapes of input data and seeds are the same
        self.seeds = seeds
        # self.seeds_inds = imma.as_seeds_inds(seeds, data.shape)

        if debug:
            logger.debug("threshold {}".format(threshold))
        if threshold is None:
            threshold = prepare_threshold_from_seeds(data=data, seeds=self.seeds, min_threshold_auto_method=threshold_auto_method)
        if debug:
            logger.debug("threshold after first evaluation {}".format(threshold))
        # import ipdb; ipdb.set_trace()
        self.threshold = threshold
        self.nObj = nObj
        self.biggestObjects = biggestObjects
        self.ICBinaryClosingIterations = binaryClosingIterations
        self.ICBinaryOpeningIterations = binaryOpeningIterations
        self.auto_method=threshold_auto_method

        self.useSeedsOfCompactObjects = useSeedsOfCompactObjects
        self.fillHoles = fillHoles

        self.threshold_upper = threshold_upper

        if (sys.version_info[0] < 3):
            import copy


            self.data = copy.copy(data)
            self.voxelsize_mm = copy.copy(voxel)

        else:
            self.data = data.copy()
            self.voxelsize_mm = voxel.copy()

        # Kalkulace objemove jednotky (voxel) (V = a*b*c)
        # voxel1 = self.voxel[0]
        # voxel2 = self.voxel[1]
        # voxel3 = self.voxel[2]
        self.voxelV = np.prod(self.voxelsize_mm, axis=None) #voxel1 * voxel2 * voxel3

        # TODO remove this nonsense
        if self.biggestObjects or (self.seeds is not None and self.useSeedsOfCompactObjects):
            self.get_priority_objects = True
        else:
            self.get_priority_objects = True
            # self.get_priority_objects = False

        self.numpyAMaxKeepDims = False

        # Pokud existuji vhodne labely, vytvori se nova data k
        # vraceni.
        # Pokud ne, vrati se "None" typ.  { Deprecated: Pokud ne,
        # vrati se cela nafiltrovana data, ktera do funkce prisla
        # (nedojde k vraceni specifickych objektu).  }


        self.firstRun = True
        if self.interactivity:
            self._init_ui(figure)

    def _init_ui(self, figure):
            if figure is None:
                self.fig = matpyplot.figure()
            else:
                self.fig = figure

            # Maximalni a minimalni pouzita hodnota prahovani v datech (bud v
            # celych datech nebo vybranych seedu)
            self.min0 = np.min(self.data)
            self.max0 = np.max(self.data)
            # self.min0 = np.amin(np.amin(self.data, axis=0))
            # if self.seeds == None:
            #
            #     self.max0 = np.amax(np.amax(self.data, axis=0))
            #     self.max0 = self.max0 + \
            #         abs(abs(self.min0) - abs(self.max0)) / 10
            #
            # else:
            #
            #     if len(self.intensities_on_seeds) > 0:
            #
            #         # Zbaveni se duplikatu.
            #         self.intensities_on_seeds = list(set(self.intensities_on_seeds))
            #         logger.debug('Hodnoty seedu: ')
            #         logger.debug(self.intensities_on_seeds)
            #
            #         self.max0 = max(self.intensities_on_seeds)
            #         self.max0 = self.max0 + \
            #             abs(abs(self.min0) - abs(self.max0)) / 10
            #
            #         # Prahy
            #         logger.debug('')
            #         logger.debug(
            #             'Minimalni doporucena hodnota prahu: ' +
            #             str(min(self.intensities_on_seeds)))
            #         logger.debug(
            #             'Maximalni doporucena hodnota prahu: ' +
            #             str(max(self.intensities_on_seeds)))
            #         logger.debug('')
            #
            #     else:
            #
            #         self.max0 = np.amax(np.amax(self.data, axis=0))
            #         self.max0 = self.max0 + \
            #             abs(abs(self.min0) - abs(self.max0)) / 10

            # Pridani subplotu do okna (do figure)
            self.ax1 = self.fig.add_subplot(222)
            self.ax2 = self.fig.add_subplot(223)
            self.ax3 = self.fig.add_subplot(224)
            self.ax4 = self.fig.add_subplot(221)

            # Upraveni subplotu
            self.fig.subplots_adjust(left=0.1, bottom=0.3)

            # Vykreslit obrazek
            self.ax1.imshow(
                np.amax(self.data, axis=0, keepdims=self.numpyAMaxKeepDims),
                self.cmap)
            self.ax2.imshow(
                np.amax(self.data, axis=1, keepdims=self.numpyAMaxKeepDims),
                self.cmap)
            self.ax3.imshow(
                np.amax(self.data, axis=2, keepdims=self.numpyAMaxKeepDims),
                self.cmap)

            # Zalozeni mist pro slidery
            left_slider_position = 0.18
            self.axcolor = 'white'  # lightgoldenrodyellow
            self.axmin = self.fig.add_axes(
                [left_slider_position, 0.24, 0.55, 0.03], axisbg=self.axcolor)
            self.axmax = self.fig.add_axes(
                [left_slider_position, 0.20, 0.55, 0.03], axisbg=self.axcolor)
            self.axclosing = self.fig.add_axes(
                [left_slider_position, 0.16, 0.55, 0.03], axisbg=self.axcolor)
            self.axopening = self.fig.add_axes(
                [left_slider_position, 0.12, 0.55, 0.03], axisbg=self.axcolor)
            self.axsigma = self.fig.add_axes(
                [left_slider_position, 0.08, 0.55, 0.03], axisbg=self.axcolor)

            # Vlastni vytvoreni slideru

            minBinaryClosing = 0
            minBinaryOpening = 0
            minSigma = 0.00


            init_thr = self.min0
            if self.threshold is not None:
                init_thr = self.threshold
            self.smin = Slider(
                self.axmin, 'Min. thr.' + str(self.min0),
                self.min0, self.max0, valinit=init_thr, dragging=True, valfmt="%4g")
                # self.min0, self.max0, valinit=self.threshold, dragging=True)
            self.smax = Slider(
                self.axmax, 'Max. thr.' + str(self.min0),
                self.min0, self.max0, valinit=self.max0, dragging=True)

            if (self.ICBinaryClosingIterations >= 1):
                self.sclose = Slider(
                    self.axclosing, 'Bin. closing', minBinaryClosing,
                    100, valinit=self.ICBinaryClosingIterations, dragging=False)
            else:
                self.sclose = Slider(
                    self.axclosing, 'Bin. closing', minBinaryClosing, 100,
                    valinit=0, dragging=False)

            if (self.ICBinaryOpeningIterations >= 1):
                self.sopen = Slider(
                    self.axopening, 'Bin. opening', minBinaryOpening,
                    100, valinit=self.ICBinaryOpeningIterations, dragging=False)
            else:
                self.sopen = Slider(
                    self.axopening, 'Bin. opening', minBinaryOpening, 100,
                    valinit=0, dragging=False)

            self.ssigma = Slider(
                self.axsigma, 'Sigma', 0.00, self.number,
                valinit=self.inputSigma, dragging=False)

            # Funkce slideru pri zmene jeho hodnoty
            self.smin.on_changed(self.updateImage)
            self.smax.on_changed(self.updateImage)
            self.sclose.on_changed(self.updateImage)
            self.sopen.on_changed(self.updateImage)
            self.ssigma.on_changed(self.updateImage)

            # Zalozeni mist pro tlacitka
            letf_button_position = 0.80
            self.axbuttprev1_5 = self.fig.add_axes(
                [letf_button_position + 0.02, 0.24, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttprev1 = self.fig.add_axes(
                [letf_button_position + 0.05, 0.24, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttnext1 = self.fig.add_axes(
                [letf_button_position + 0.09, 0.24, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttnext1_5 = self.fig.add_axes(
                [letf_button_position + 0.12, 0.24, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttprev2_5 = self.fig.add_axes(
                [letf_button_position + 0.02, 0.20, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttprev2 = self.fig.add_axes(
                [letf_button_position + 0.05, 0.20, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttnext2 = self.fig.add_axes(
                [letf_button_position + 0.09, 0.20, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttnext2_5 = self.fig.add_axes(
                [letf_button_position + 0.12, 0.20, 0.03, 0.035], axisbg=self.axcolor)
            self.axbuttnextclosing = self.fig.add_axes(
                [letf_button_position + 0.09, 0.16, 0.05, 0.035], axisbg=self.axcolor)
            self.axbuttprevclosing = self.fig.add_axes(
                [letf_button_position + 0.02, 0.16, 0.05, 0.035], axisbg=self.axcolor)
            self.axbuttnextopening = self.fig.add_axes(
                [letf_button_position + 0.09, 0.12, 0.05, 0.035], axisbg=self.axcolor)
            self.axbuttprevopening = self.fig.add_axes(
                [letf_button_position + 0.02, 0.12, 0.05, 0.035], axisbg=self.axcolor)
            self.axbuttreset = self.fig.add_axes(
                [letf_button_position + 0.01, 0.07, 0.08, 0.045], axisbg=self.axcolor)
            self.axbuttcontinue = self.fig.add_axes(
                [letf_button_position + 0.09, 0.07, 0.08, 0.045], axisbg=self.axcolor)

            # Zalozeni tlacitek
            self.bnext1 = Button(self.axbuttnext1, '+1')
            self.bnext1_5 = Button(self.axbuttnext1_5, '+5')
            self.bprev1 = Button(self.axbuttprev1, '-1')
            self.bprev1_5 = Button(self.axbuttprev1_5, '-5')
            self.bnext2 = Button(self.axbuttnext2, '+1')
            self.bnext2_5 = Button(self.axbuttnext2_5, '+5')
            self.bprev2 = Button(self.axbuttprev2, '-1')
            self.bprev2_5 = Button(self.axbuttprev2_5, '+5')
            self.bnextclosing = Button(self.axbuttnextclosing, '+1.0')
            self.bprevclosing = Button(self.axbuttprevclosing, '-1.0')
            self.bnextopening = Button(self.axbuttnextopening, '+1.0')
            self.bprevopening = Button(self.axbuttprevopening, '-1.0')
            self.breset = Button(self.axbuttreset, 'Reset')
            self.bcontinue = Button(self.axbuttcontinue, 'Done')

            # Funkce tlacitek pri jejich aktivaci
            self.bnext1.on_clicked(self.buttonMinNext)
            self.bprev1.on_clicked(self.buttonMinPrev)
            self.bnext2.on_clicked(self.buttonMaxNext)
            self.bprev2.on_clicked(self.buttonMaxPrev)
            self.bnext1_5.on_clicked(self.buttonMinNext5)
            self.bprev1_5.on_clicked(self.buttonMinPrev5)
            self.bnext2_5.on_clicked(self.buttonMaxNext5)
            self.bprev2_5.on_clicked(self.buttonMaxPrev5)
            self.bnextclosing.on_clicked(self.buttonNextClosing)
            self.bprevclosing.on_clicked(self.buttonPrevClosing)
            self.bnextopening.on_clicked(self.buttonNextOpening)
            self.bprevopening.on_clicked(self.buttonPrevOpening)
            self.breset.on_clicked(self.buttonReset)
            self.bcontinue.on_clicked(self.buttonContinue)

            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.smax.valtext.set_text('{}'.format(self.smax.val))
            logger.debug("self.threshold at the end of init(): " + str(self.threshold))

    def run(self):
        """

            Spusteni UI.

        """

        if(self.errorsOccured == True):

            return self.data

        self.firstRun = True

        self.updateImage(-1)
        garbage.collect()
        if self.interactivity:
            self.on_show_fcn()

        del(self.data)
        garbage.collect()
        return self.imgFiltering

    def returnLastThreshold(self):
        return self.threshold

    def updateImage(self, val):
        """
        Hlavni update metoda.
        Cinny kod pro gaussovske filtrovani, prahovani, binarni uzavreni a
        otevreni a vraceni nejvetsich nebo oznacenych objektu.

        """

        # import ipdb
        # ipdb.set_trace()

        # Filtrovani

        # Zjisteni jakou sigmu pouzit
        if(self.firstRun == True and self.inputSigma >= 0):
            sigma = np.round(self.inputSigma, 2)
        elif self.interactivity:
            sigma = np.round(self.ssigma.val, 2)
        else:
            sigma = np.round(self.inputSigma, 2)


        # Prahovani (smin, smax)

        # max_threshold = self.threshold_upper
        # min_threshold = self.threshold

        if self.interactivity:

            self.smin.val = (np.round(self.smin.val, 2))
            self.smin.valtext.set_text('{}'.format(self.smin.val))
            self.smax.val = (np.round(self.smax.val, 2))
            self.smax.valtext.set_text('{}'.format(self.smax.val))

            self.threshold = self.smin.val
            self.threshold_upper = self.smax.val

            closeNum = int(np.round(self.sclose.val, 0))
            openNum = int(np.round(self.sopen.val, 0))
            self.sclose.valtext.set_text('{}'.format(closeNum))
            self.sopen.valtext.set_text('{}'.format(openNum))

        else:

            closeNum = self.ICBinaryClosingIterations
            openNum = self.ICBinaryOpeningIterations

        # make_image_processing(sigma, min_threshold, max_threshold, closeNum, openNum, auto_method=self.)
        self.imgFiltering, self.threshold = make_image_processing(data=self.data, voxelsize_mm=self.voxelsize_mm,
                                                                  seeds=self.seeds, sigma_mm=sigma,
                                                                  min_threshold=self.threshold,
                                                                  max_threshold=self.threshold_upper, closeNum=closeNum,
                                                                  openNum=openNum,
                                                                  min_threshold_auto_method=self.auto_method,
                                                                  fill_holes=self.fillHoles,
                                                                  get_priority_objects=self.get_priority_objects,
                                                                  nObj=self.nObj)
        # Vykresleni dat
        if (self.interactivity == True):
            self.drawVisualization()

        # Nastaveni kontrolnich hodnot
        self.firstRun = False

        garbage.collect()

        self.debugInfo()


    def debugInfo(self):

        logger.debug('======')
        logger.debug('!Debug')
        logger.debug('\tUpdate cycle:')
        logger.debug('\t\tThreshold min: ' +
                     str(self.threshold))
        logger.debug('\t\tThreshold max: ' +
                     str(self.threshold_upper))
        if (self.interactivity == True):
            logger.debug(
                '\t\tThreshold max: ' + str(self.smax.val))
            logger.debug(
                '\t\tBinary closing: ' + str(np.round(self.sclose.val, 0)))
            logger.debug(
                '\t\tBinary opening: ' + str(np.round(self.sopen.val, 0)))
            logger.debug(
                '\t\tSigma filter param: ' + str(np.round(self.ssigma.val,
                                                             2)))
        logger.debug('======')

    def getBiggestObjects(self):
        """

        Vraceni nejvetsich objektu (nebo objektu, ktere obsahuji prioritni
        seedy).

        """

        logger.debug('biggest objects ' + str(self.biggestObjects))
        logger.debug('self.seeds ' + str(self.seeds))


    def __drawSegmentedSlice(self, ax, contour, i):
        """
        Used for visualization of midle slice of the data3d
        :param ax:
        :param contour:
        :param i:
        :return:
        """
        i = int(i)
        ax.cla()
        ax.imshow(self.data[i, :, :], cmap=self.cmap)

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        if contour is not None:
            #ax.contour(contour[i, :, :])
            logger.debug("type contour " + str(type(contour)))
            logger.debug("max contour " + str(np.max(contour)))
            logger.debug("min contour " + str(np.min(contour)))
            logger.debug("type contour " + str(type(contour[i,:,:])))
            logger.debug("max contour " + str(np.max(contour[i,:,:])))
            logger.debug("min contour " + str(np.min(contour[i,:,:])))
            contour_slice = contour[i, :, :]
            if np.max(contour_slice) != np.min(contour_slice):
                ax.contour(contour[i, :, :] + 1)

    def drawVisualization(self):
        """

        Vykresleni dat.

        """

        # Predani dat k vykresleni
        if (self.imgFiltering is None):

            # print '(DEBUG) Typ dat: ' + str(type(self.data[0][0][0]))

            self.ax1.imshow(
                np.amax(np.zeros(self.data.shape), axis=0,
                           keepdims=self.numpyAMaxKeepDims), self.cmap)
            self.ax2.imshow(
                np.amax(np.zeros(self.data.shape), axis=1,
                           keepdims=self.numpyAMaxKeepDims), self.cmap)
            self.ax3.imshow(
                np.amax(np.zeros(self.data.shape), axis=2,
                           keepdims=self.numpyAMaxKeepDims), self.cmap)

        else:

            img0, img1, img2 = thresholding_functions.prepareVisualization(
                self.imgFiltering)

            # t2 = time.time()
            # print 't1 %f t2 %f ' % (t1 - t0, t2 - t1)

            self.ax1.imshow(img0, self.cmap)
            self.ax2.imshow(img1, self.cmap)
            self.ax3.imshow(img2, self.cmap)

            # del(img0)
            # del(img1)
            # del(img2)

        self.ax1.set_xticklabels([])
        self.ax1.set_yticklabels([])
        self.ax2.set_xticklabels([])
        self.ax2.set_yticklabels([])
        self.ax3.set_xticklabels([])
        self.ax3.set_yticklabels([])
        self.ax4.set_xticklabels([])
        self.ax4.set_yticklabels([])
        if self.imgFiltering is not None:
            self.__drawSegmentedSlice(self.ax4, self.imgFiltering, int(self.imgFiltering.shape[0] / 2))
        # Prekresleni
        self.fig.canvas.draw()

    def buttonReset(self, event):

        self.sclose.valtext.set_text(
            '{}'.format(self.ICBinaryClosingIterations))
        self.sopen.valtext.set_text(
            '{}'.format(self.ICBinaryOpeningIterations))
        self.ssigma.valtext.set_text('{}'.format(self.inputSigma))

        self.firstRun = True
        self.lastSigma = -1
        self.threshold = -1

        self.updateImage(0)

    def buttonContinue(self, event):

        matpyplot.clf()
        if self.on_close_fcn is not None:
            self.on_close_fcn(self)
        matpyplot.close()

    def buttonMinNext(self, event):
        self.buttonMinUpdate(event, +1.0)

    def buttonMinPrev(self, event):
        self.buttonMinUpdate(event, -1.0)

    def buttonMinNext5(self, event):
        self.buttonMinUpdate(event, +5.0)

    def buttonMinPrev5(self, event):
        self.buttonMinUpdate(event, -5.0)


    def buttonMaxNext(self, event):
        self.buttonMaxUpdate(event, -1.0)

        # if self.max0 < (self.smax.val + 1.0):
        #
        #     self.smax.val = self.max0
        #
        # else:
        #
        #     self.smax.val += 1.0
        #
        # self.smax.val = (np.round(self.smax.val, 2))
        # self.smax.valtext.set_text('{}'.format(self.smax.val))
        # self.fig.canvas.draw()
        # self.updateImage(0)

    def buttonMaxNext5 (self, event):
        self.buttonMaxUpdate(event, 5.0)
    def buttonMaxPrev5 (self, event):
        self.buttonMaxUpdate(event, -5.0)
    def buttonMaxPrev(self, event):
        self.buttonMaxUpdate(event, -1.0)

    def buttonMinUpdate(self, event, value):

        if self.min0 > (self.smin.val + value):
            self.smin.val = self.min0
        elif self.max0 < (self.smin.val + value):
            self.smin.val = self.max0
        else:
            self.smin.val += value

        self.smin.val = (np.round(self.smin.val, 2))
        self.smin.valtext.set_text('{}'.format(self.smin.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonMaxUpdate(self, event, value):

        if self.min0 > (self.smax.val + value):
            self.smax.val = self.min0
        elif self.max0 < (self.smax.val + value):
            self.smax.val = self.max0
        else:
            self.smax.val += value

        self.smax.val = (np.round(self.smax.val, 2))
        self.smax.valtext.set_text('{}'.format(self.smax.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonNextOpening(self, event):

        self.sopen.val += 1.0
        self.sopen.val = (np.round(self.sopen.val, 2))
        self.sopen.valtext.set_text('{}'.format(self.sopen.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrevOpening(self, event):

        if(self.sopen.val >= 1.0):
            self.sopen.val -= 1.0
            self.sopen.val = (np.round(self.sopen.val, 2))
            self.sopen.valtext.set_text('{}'.format(self.sopen.val))
            self.fig.canvas.draw()
            self.updateImage(0)

    def buttonNextClosing(self, event):

        self.sclose.val += 1.0
        self.sclose.val = (np.round(self.sclose.val, 2))
        self.sclose.valtext.set_text('{}'.format(self.sclose.val))
        self.fig.canvas.draw()
        self.updateImage(0)

    def buttonPrevClosing(self, event):

        if(self.sclose.val >= 1.0):
            self.sclose.val -= 1.0
            self.sclose.val = (np.round(self.sclose.val, 2))
            self.sclose.valtext.set_text('{}'.format(self.sclose.val))
            self.fig.canvas.draw()
            self.updateImage(0)

def prepare_threshold_from_seeds(data, seeds, min_threshold_auto_method):

    if seeds is not None:
        intensities_on_seeds = thresholding_functions.get_intensities_on_seed_position(data, seeds == 1)
    else:
        intensities_on_seeds = None
    logger.debug("intensities on seeds {}".format(intensities_on_seeds))
    if min_threshold_auto_method is 'otsu':
        logger.debug('using otsu threshold')
        min_threshold = thresholding_functions.calculateAutomaticThresholdOtsu(
            data, intensities_on_seeds)
    else:
        min_threshold = thresholding_functions.calculateAutomaticThreshold(
            data, intensities_on_seeds)
    logger.debug("min threshold prepared {}".format(intensities_on_seeds))
    return min_threshold


def make_image_processing(
        data, voxelsize_mm, seeds=None, sigma_mm=1, min_threshold=None, max_threshold=None,
        closeNum=0, openNum=0, min_threshold_auto_method="", fill_holes=True,
        get_priority_objects=True, nObj=1, debug=False):
    if (sys.version_info[0] < 3):
        import copy


        data_copy = copy.copy(data)
    else:
        data_copy = data.copy()


    if sigma_mm > 0:
        sigmaNew = thresholding_functions.calculateSigma(voxelsize_mm, sigma_mm)
        data_copy = thresholding_functions.gaussFilter(
            data_copy, sigmaNew)

        del(sigmaNew)
    if debug:
        import sed3


        # ed = sed3.sed3qt(data_copy)
        sed3.show_slices(data_copy, shape=[6, 9])
        # ed.show()

    if min_threshold is None:
        min_threshold = prepare_threshold_from_seeds(data=data_copy, seeds=seeds,
                                                     min_threshold_auto_method=min_threshold_auto_method)

    data_thr = thresholding_functions.thresholding(
        data_copy,
        min_threshold,
        max_threshold,
        use_min_threshold=True,
        use_max_threshold=max_threshold is not None
    )
    if debug:
        logger.debug("np min median max input data "
                     + str(np.min(data_copy)) + " "
                     + str(np.median(data_copy)) + " "
                     + str(np.max(data_copy)) + " "
                     )

    if debug:
        logger.debug("np unique sum binar "
                     + str(np.unique(data_thr)) + " "
                     + str(np.sum(data_thr)) + " "
                     )

    data_thr = thresholding_functions.binaryClosingOpening(
        data_thr, closeNum, openNum, True)

    # Fill holes
    if fill_holes:

        data_thr = thresholding_functions.fillHoles(
            data_thr)

    # use a wall for label 2
    if type(seeds) is np.ndarray:
        data_thr[seeds==2] = 0
    # Zjisteni nejvetsich objektu.
    if get_priority_objects:
        if seeds is not None:
            selected_seeds = seeds == 1
        else:
            selected_seeds = seeds
        data_thr = thresholding_functions.get_priority_objects(
            data_thr, nObj, selected_seeds
        )

    if debug:
        logger.debug("np unique sum binar hist end "
                     + str(np.unique(data_thr)) + " "
                     + str(np.sum(data_thr)) + " "
                     # + str(np.histogram(data_thr, bins="auto")) + " "
                     )

    return data_thr, min_threshold


def main():
    import numpy as np


    data = np.random.randint(0, 30, [15, 16, 18])
    print(data.shape)
    data[5:11, 7:13, 2:10] += 20
    uit = uiThreshold(data=data, voxel=[1, 2, 1.5])
    uit.run()


if __name__ == "__main__":
    main()

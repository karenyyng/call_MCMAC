"""
Author : Karen Ng
This program takes the output pickle arrays of TSM.py and creates some plots
and statistics.  This is largely based on PlotTSM.py or PlotTSM_array.py by
Will Dawson.

The main function of this program is to generate report quality plots,
especially a covariance array plot
"""
from __future__ import division
import pylab
import numpy as np
import numpy
import pickle
import warnings
from astrostats import biweightLoc, bcpcl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import pandas as pd
from astropy.cosmology import FlatLambdaCDM
from astroML.plotting import hist as astroMLhist

warnings.filterwarnings("ignore", category=DeprecationWarning)
from astroML import density_estimation as de

# -------- for masking and subsetting --------------------------------------
def loadcombo(prefix, index, suffix):
    """loads the data from pickle files
    Parameters:
    ===========
    prefix = string
        denotes the name of the path to the file
    index = number
        denotes the index to add to the file name
    filename consists of prefix + suffix+.pickle see below

    Returns
    =======
    numpy array
        data
    """
    array = []
    for i in index:
        filename = prefix + i + '_' + suffix + '.pickle'
        # read in the pickled array
        F = open(filename)
        tmp = pickle.load(F)
        F.close()
        array = numpy.append(array, tmp)

    #filename = prefix+suffix+'.pickle'
    # read in the pickled array
    #F = open(filename)
    #tmp = pickle.load(F)
    # F.close()
    #array = numpy.append(array,tmp)
    return array


def load_pickles_to_df(par, prefix, index, msun1e14=True,
                       msun_string=['m_1', 'm_2'], verbose=False):
    """
    Parameters
    =========
    par = list of strings
        denotes the name of the parameters
    prefix = string
        prefix that denotes the path and file name excluding extension
        .pickle
    index = number
        how many pickle files there are
    msun1e14 = logical
        if we want units of mass to be in 1e14 M_sun
    msun_string = list of strings
        the strings are the keys to the df for changing the units
    verbose: logical

    Returns
    ======
    Pandas Dataframe that have all nans removed

    """
    for i in range(len(par)):
        if(verbose):
            print 'loading ' + par[i]
        d = loadcombo(prefix, index, par[i])
        if i == 0:
            data = pd.DataFrame(d, columns=[par[i]])
        else:
            data[par[i]] = pd.DataFrame(d)

    print "dropping NA!"
    data = data.dropna()

    if msun1e14 is True:
        for mass in msun_string:
            print "converting entry " + mass + \
                " to units of 1e14 m_sun"
            data[mass] = data[mass] / (1e14)

    return data


def mask_bigger_than_age_of_universe(
    df, z, H0, Om0, T=None, TSM_0=None, TSM_1=None):
    """assumes a FlatLambdaCDM cosmology to calculate age of universe at
    particular redshift and returns a mask that masks time values
    that are bigger than the age of the universe

    Parameters
    =========
    df = pandas dataframe - outputs from load_pickles_to_df
    z = float - redshift
    H0 = Hubble parameter
    Om0 = Relative density of matter
    T = string - denotes column name for T in the dataframe
    TSM_0 = string - denotes column name for TSM_0 in the dataframe
    TSM_1 = string - denotes column name for TSM_1 in the dataframe

    Returns
    ======
    combined mask = numpy array
    age = float - age of the universe at corresponding redshift in Gyrs
    """
    assert T is not None or TSM_0 is not None or TSM_1 is not None, \
        "no relevant col names for T, TSM_0 and TSM_1 specified"

    cosmology = FlatLambdaCDM(H0=H0, Om0=Om0)
    age_of_universe = cosmology.age(z)
    age = age_of_universe.value

    masks = {}
    if TSM_0 is not None:
        masks[TSM_0] = df[TSM_0] < age
    if TSM_1 is not None:
        masks[TSM_1] = df[TSM_1] < age
    if T is not None:
        masks[T] = df[T] < age

    mask = np.ones(df.shape[0])
    for m in masks.values():
        mask = np.logical_and(mask, m)
    print "# of masked rows = {0}".format(df.shape[0] - np.sum(mask))
    print "# of remaining rows = {0}".format(np.sum(mask))
    print "% of original data remaining = {0:.2f}".format(
        (np.sum(mask)) / df.shape[0] * 100)

    return mask, age


def radio_dist_prior(d_3D, d_3Dmax=3.0, d_3Dmin=1.0):
    '''
    Stability: to be tested
    input:
    d_3D = numpy array to be masked, in unit of Mpc
    d_3Dmax = float, the upper limit to be masked out, in unit of Mpc
    d_3Dmin = float, the lower limit to be masked out, in unit of Mpc
    output:
    mask = numpy array that gives 1 if it is within the range
            0 if it is NOT within the specified range
    count = number of entries along the array that has value 1
    '''
    mask = np.logical_and(d_3D < d_3Dmax, d_3D >= d_3Dmin)
    count = np.sum(mask)
    return mask, count


def radio_polar_prior(alpha, alpha_min=0., alpha_max=40):
    '''
    Stability: to be tested
    input:
    alpha = numpy array to be masked, in units of degrees
    alpha_min = float, if alpha is smaller than this value it's masked out
    alpha_max = float, if alpha is bigger than this value, it's masked out
    output:
    mask = numpy array that gives 1 if it is within the range and 0
            otherwise
    '''
    mask = np.logical_and(alpha < alpha_max, alpha > alpha_min)
    count = np.sum(mask)
    print "# of masked rows = {0}".format(alpha.size - count)
    print "# of remaining rows = {0}".format(count)
    print "% of original data remaining = {0:.2f}".format(
        count / alpha.size * 100.)
    return mask


def apply_radioprior(radiomask, dataarray):
    '''
    Checks if the length data array is the same as the prior mask
    if not, do nothing
    if lengths are the same, apply the prior
    starts examine if the length of mask is the same
    as the length of the array to be masked
    input:
    mask = numpy array with true or false as the values
    dataarray = numpy data array to be masked
    '''
    if len(radiomask) != len(dataarray):
        print 'length of mask and data array does not match!'
        print 'skipping the application of radio relic prior'
    else:
        # apply the mask
        temp = dataarray * radiomask

        counter = 0
        # removed the entries that are zero from the data array
        # to avoid the zero entries being binned
        for n in range(len(temp)):
            if temp[n] != 0.0:
                counter += 1
        # print 'number of non-zero entries after masking is', counter

        dataarray = numpy.zeros(counter)
        ncounter = 0
        for n in range(len(temp)):
            if temp[n] != 0.0:
                dataarray[ncounter] = temp[n]
                ncounter += 1
        # print 'number of non-zero entries for data array after masking is',
        # ncounter

    return dataarray


# --------modified version of  Will 's original functions----------------------
def histplot1d_pdf(x, prefix, prob=None, N_bins='knuth', histrange=None,
                   x_lim=None, y_lim=None, x_label=None, y_label=None,
                   legend=None, title=None, save=True, verbose=True,
                   plot=False):
    """plot data as normalized pdf
    plot the pdf of the histograms
    might want to return the pdf later on

    x = numpy array like object, can be dataframe columns
        data to be plotted on the x-axis
    prefix = string
        denotes the output file prefix
    prob = numpy array with same size as x
        denotes the weight to be put for correcting bias
    N_bins = integer
        denotes the number of bins
    histrange = a size 2 numpy array / list
        denotes the lower and upper range for making the histogram
    x_lim = a size 2 numpy array
    y_lim = a size 2 numpy array
    x_label = string
    y_label = string
    legend = string
    title = string
    """
    # compare bin width to knuth bin width
    binwidth, bins = de.knuth_bin_width(x, return_bins=True)
    knuth_N_bins = bins.size - 1
    if type(N_bins) is int:
        print "Specified bin width is {0}, Knuth bin size is {1}".format(
            N_bins, knuth_N_bins)
    elif N_bins == 'knuth':
        print "Knuth bin no is {0}".format(knuth_N_bins)
        N_bins = knuth_N_bins

    hist, binedges, tmp = \
        astroMLhist(x, bins=N_bins, histtype='step',
                   weights=prob, range=histrange, color='k', linewidth=2)

    # do not want to plot the graph without normalization
    # but the output of the binned array is needed for calculation
    # for location and confidence levels below
    pylab.close()

    fig = pylab.figure()
    # plot the normalized version (pdf) of the data
    pdf, binedges, holder = astroMLhist(x, bins=N_bins, histtype='step',
                                        weights=prob, range=histrange,
                                        color='k', linewidth=2, normed=1)

    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted
    # data I need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned = \
                numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        else:
            x_temp = numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
            x_binned = numpy.concatenate((x_binned, x_temp))

    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc, x_binned, 1)
    ll_95, ul_95 = bcpcl(loc, x_binned, 2)
    results = [loc, ll_68, ul_68, ll_95, ul_95]

    if verbose is True:
        print '{0}, {1:0.4f}, {2:0.4f},'.format(prefix, loc, ll_68) + \
            '{0:0.4f}, {1:0.4f}, {2:0.4f}'.format(ul_68, ll_95, ul_95)

    if save is not True:
        return results

    # Create location and confidence interval line plots
    pylab.axvline(loc, ls='--', linewidth=2, label='$C_{BI}$', color="k")
    pylab.axvline(ll_68, ls='-.', linewidth=2, color='#800000',
                  label='68% $IC_{B_{BI}}$')
    pylab.axvline(ul_68, ls='-.', linewidth=2, color='#800000')
    pylab.axvline(ll_95, ls=':', linewidth=2, color='#0000A0',
                  label='95% $IC_{B_{BI}}$')
    pylab.axvline(ul_95, ls=':', linewidth=2, color='#0000A0')

    if x_label is not None:
        pylab.xlabel(x_label, fontsize=15)
    if y_label is not None:
        pylab.ylabel(y_label, fontsize=15)
    if x_lim is not None:
        pylab.xlim(x_lim)
    if y_lim is not None:
        pylab.ylim(y_lim)
    if legend is not None:
        pylab.legend()
    if title is not None:
        pylab.title(title, fontsize = 16)

    ### set font size
    # fontsize=14
    # ax = pylab.gca()
    # for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)

    filename = prefix + '_histplot1D'
    pylab.savefig(filename + '.pdf', bbox_inches='tight')

    return results


def histplot1d(x, prefix=None, prob=None, norm=False, N_bins='knuth',
               histrange=None, x_lim=None, y_lim=None, x_label=None,
               y_label=None, legend=None, save=False, verbose=False):
    """summarize the CI of the data
    plots data after binning and weighting the data appropriately
    """

    # compare bin width to knuth bin width
    binwidth, bins = de.knuth_bin_width(x, return_bins=True)
    knuth_N_bins = bins.size - 1
    if type(N_bins) is int:
        print "specified bin width is {0}, Knuth bin size is {1}".format(
            N_bins, knuth_N_bins)
    elif N_bins == 'knuth':
        N_bins = knuth_N_bins

    fig = pylab.figure()
    hist, binedges, tmp = astroMLhist(
        x, bins=N_bins, histtype='step', weights=prob, range=histrange,
        color='k', linewidth=2, normed=norm)
    pylab.close()

    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations

    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
        else:
            x_temp = numpy.ones(hist[i])*(binedges[i]+binedges[i+1])/2
            x_binned = numpy.concatenate((x_binned, x_temp))

    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc, x_binned, 1)
    ll_95, ul_95 = bcpcl(loc, x_binned, 2)
    results = [loc, ll_68, ul_68, ll_95, ul_95]
    if save is not True:
        return results

    assert prefix is not None, "prefix cannot be None"

    # Create location and confidence interval line plots
    # find the binedge that the location falls into
    # so that the line indicating the location only extends to top of
    # histogram
    loc_ix = find_bin_ix(binedges, loc)
    ll_68_ix = find_bin_ix(binedges, ll_68)
    ul_68_ix = find_bin_ix(binedges, ul_68)
    ll_95_ix = find_bin_ix(binedges, ll_95)
    ul_95_ix = find_bin_ix(binedges, ul_95)

    ax.plot((loc, loc), (0, hist[loc_ix - 1]), ls='--', lw=1, color="k")

    width = binedges[ll_68_ix + 1] - binedges[ll_68_ix]
    for i in range(ll_68_ix, ul_68_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.6)
    for i in range(ll_95_ix, ul_95_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.3)

    if x_label != None:
        pylab.xlabel(x_label, fontsize=20)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=20)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        pylab.legend()
    # fontsize=14
    #ax = pylab.gca()
    # for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)

    filename = prefix+'_histplot1D'
    if save:
        pylab.savefig(filename, dpi=300, bbox_inches='tight')

    if verbose:
        print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(
            prefix, loc, ll_68, ul_68, ll_95, ul_95)

    return results


def histplot1d_part(ax, x, prob=None, N_bins='knuth', histrange=None,
                    x_lim=None, y_lim=None):
    '''
    This take the additional value of an array axes. for use with subplots
    similar to histplot1d but for subplot purposes I believe
    '''
    # compare bin width to knuth bin width
    binwidth, bins = de.knuth_bin_width(x, return_bins=True)
    knuth_N_bins = bins.size - 1
    if type(N_bins) is int:
        print "specified bin width is {0}, Knuth bin size is {1}".format(
            N_bins, knuth_N_bins)
    elif N_bins == 'knuth':
        N_bins = knuth_N_bins

    hist, binedges, tmp = ax.hist(
        x, bins=N_bins, histtype='step', weights=prob, range=histrange,
        color='k', linewidth=1)

    # Calculate the location and %confidence intervals
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned = \
                numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        elif numpy.size(x_binned) == 0:
            x_binned = \
                numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
        else:
            x_temp = \
                numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
            x_binned = numpy.concatenate((x_binned, x_temp))
    loc = biweightLoc(x_binned)
    ll_68, ul_68 = bcpcl(loc, x_binned, 1)
    ll_95, ul_95 = bcpcl(loc, x_binned, 2)

    # Create location and confidence interval line plots
    # find the binedge that the location falls into
    # so that the line indicating the location only extends to top of
    # histogram
    loc_ix = find_bin_ix(binedges, loc)
    ll_68_ix = find_bin_ix(binedges, ll_68)
    ul_68_ix = find_bin_ix(binedges, ul_68)
    ll_95_ix = find_bin_ix(binedges, ll_95)
    ul_95_ix = find_bin_ix(binedges, ul_95)

    ax.plot((loc, loc), (0, hist[loc_ix - 1]), ls='--', lw=1, color="k")

    width = binedges[ll_68_ix + 1] - binedges[ll_68_ix]
    for i in range(ll_68_ix, ul_68_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.6)
    for i in range(ll_95_ix, ul_95_ix):
        ax.bar(binedges[i], hist[i], width, lw=0, color="b", alpha=.3)

    if x_lim != None:
        ax.set_xlim(x_lim)
    if y_lim != None:
        ax.set_ylim(y_lim)
    return loc, ll_68, ul_68, ll_95, ul_95


def find_bin_ix(binedges, loc):
    """find the index in the numpy array binedges that corresponds to loc"""
    find_loc_i = binedges < loc
    return np.sum(find_loc_i)


def comb_zip(ls1, ls2):
    return [(lb1, lb2) for lb1 in ls1 for lb2 in ls2]


def histplot2d(x, y, prefix, prob=None, N_bins=100, histrange=None,
               x_lim=None, y_lim=None, x_label=None, y_label=None,
               legend=None, save=False):
    '''
    Input:
    plot 2d histogram of 2 data arrays
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    # prevent masked array from choking up the 2d histogram function
    x = np.array(x)
    y = np.array(y)
    # Create the confidence interval plot
    if histrange == None:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins=N_bins, weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x, y, bins=N_bins)
    else:
        if prob != None:
            H, xedges, yedges = \
                numpy.histogram2d(x, y, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                         [histrange[2], histrange[3]]],
                                  weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins = N_bins,
                range=[[histrange[0], histrange[1]],
                       [histrange[2], histrange[3]]])
    H = numpy.transpose(H)
    # Flatten H
    h = numpy.reshape(H, (N_bins**2))
    # Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum / h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum / h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    y = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    X, Y = numpy.meshgrid(x, y)

    fig = pylab.figure()
    # Countours
    CS = pylab.contour(X, Y, H, (h_2sigma, h_1sigma), linewidths=(2, 2))
    # imshow
    #im = pylab.imshow(H,cmap=pylab.cm.gray)
    pylab.pcolor(X, Y, H, cmap=pylab.cm.gray_r)

    if x_label != None:
        pylab.xlabel(x_label, fontsize=14)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=14)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        # Dummy lines for legend
        # 800000 is for Maroon color - 68% 1 sigma
        # 0000A0 is for blue color - 95% confidence 3 sigma
        pylab.plot((0, 1), (0, 1), c='#800000', linewidth=2, label=('68%'))
        pylab.plot((0, 1), (0, 1), c='#0000A0', linewidth=2, label=('95%'))
        pylab.legend(scatterpoints=1)
    fontsize = 15
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    if save:
        filename = prefix+'_histplot2d'
        pylab.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def histplot2d_part(ax, x, y, prob=None, N_bins=100, histrange=None,
                    x_lim=None, y_lim=None):
    '''
    similar to histplot2d
    This take the additional value of an array axes. for use with subplots
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    # prevent masked array from choking up the 2d histogram function
    x = np.array(x)
    y = np.array(y)

    # Create the confidence interval plot
    assert prob is not None, "there is no prob given for weighting"

    if histrange is None:
        if prob is not None:
            H, xedges, yedges = \
                numpy.histogram2d(x, y, bins=N_bins, weights=prob)
        elif prob is None:
            H, xedges, yedges = numpy.histogram2d(x, y, bins=N_bins)
    else:
        if prob is not None:
            H, xedges, yedges = \
                numpy.histogram2d(x, y, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                        [histrange[2], histrange[3]]],
                                  weights=prob)
        elif prob is None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins=N_bins, range=[[histrange[0], histrange[1]],
                                          [histrange[2], histrange[3]]])
    H = numpy.transpose(H)
    # Flatten H
    h = numpy.reshape(H, (N_bins ** 2))
    # Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum / h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum / h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1] + (xedges[1] - xedges[0]) / 2
    y = yedges[:-1] + (yedges[1] - yedges[0]) / 2
    X, Y = numpy.meshgrid(x, y)

    # Countours
    CS = ax.contour(X, Y, H, (h_2sigma, h_1sigma), linewidths=(2, 2),
                    colors=((158 / 255., 202 / 255., 225 / 255.),
                            (49 / 255., 130 / 255., 189 / 255.)))
    # imshow
    #im = ax.imshow(H,cmap=ax.cm.gray)
    ax.pcolor(X, Y, H, cmap=pylab.cm.gray_r)

    if x_lim is not None:
        ax.set_xlim(x_lim)
    if y_lim is not None:
        ax.set_ylim(y_lim)


def histplot2dTSC(x, y, prefix, prob=None, N_bins=100, histrange=None,
                  x_lim=None, y_lim=None, x_label=None, y_label=None,
                  legend=None):
    '''
    this is the one for generating the 2d plot in will 's paper?...
    Input:
    x = [1D array of N floats]
    y = [1D array of N floats]
    prefix = [string] prefix of output file
    prob = [None] or [1D array of N floats] weights to apply to each (x,y) pair
    N_bins = [integer] the number of bins in the x and y directions
    histrange = [None] or [array of floats: (x_min,x_max,y_min,y_max)] the range
        over which to perform the 2D histogram and estimate the confidence
        intervals
    x_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    y_lim = [None] or [array of floats: (x_min,x_max)] min and max of the range
        to plot
    x_label = [None] or [string] the plot's x-axis label
    y_label = [None] or [string] the plot's y-axis label
    legend = [None] or [True] whether to display a legend or not
    '''
    # Input calculated v and t parameters for other Dissociative Mergers
    v_bullet_analytic = 3400
    t_bullet_analytic = 0.218

    v_bullet_sf07 = 3400
    t_bullet_sf07 = 0.18

    v_macs = 2000
    t_macs = 0.255

    v_a520 = 2300
    t_a520 = 0.24

    v_pandora = 4045
    t_pandora = 0.162

    # Create the confidence interval plot
    if histrange is None:
        if prob is not None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins=N_bins, weights=prob)
        elif prob is None:
            H, xedges, yedges = numpy.histogram2d(x, y, bins=N_bins)
    else:
        if prob is not None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins=N_bins,
                range=[[histrange[0], histrange[1]],
                      [histrange[2], histrange[3]]],
                weights=prob)
        elif prob is None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins=N_bins, range=[[histrange[0], histrange[1]],
                                          [histrange[2], histrange[3]]])
    H = numpy.transpose(H)
    # Flatten H
    h = numpy.reshape(H, (N_bins ** 2))
    # Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum/h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum/h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1]+(xedges[1]-xedges[0])/2
    y = yedges[:-1]+(yedges[1]-yedges[0])/2
    X, Y = numpy.meshgrid(x, y)

    fig = pylab.figure()
    # Countours
    CS = pylab.contour(X, Y, H, (h_2sigma, h_1sigma), linewidths=(2, 2))
    # imshow
    #im = pylab.imshow(H,cmap=pylab.cm.gray)
    pylab.pcolor(X, Y, H, cmap=pylab.cm.gray_r)

    # Data points for other dissociative mergers
    pylab.scatter(v_bullet_sf07, t_bullet_sf07, s=140,
                  c='k', marker='d', label="Bullet SF07")
    #pylab.scatter(v_macs,t_macs,s=140, c='0.4',markeredgecolor='0.4', marker='^',label='MACS J0025.4')
    # pylab.scatter(v_a520,t_a520,s=140,c='0.4',markeredgecolor='0.4',marker='o',label='A520')
    # pylab.scatter(v_pandora,t_pandora,s=140,c='0.4',markeredgecolor='0.4',marker='p',label='A2744')

    if x_label != None:
        pylab.xlabel(x_label, fontsize=20)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=20)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        # Dummy lines for legend
        pylab.plot((0, 1), (0, 1), c='#800000', linewidth=2, label=('68%'))
        pylab.plot((0, 1), (0, 1), c='#0000A0', linewidth=2, label=('95%'))
        pylab.legend(scatterpoints=1)
    fontsize = 20
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    filename = prefix+'_histplot2dTSC.pdf'
    pylab.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def percentdiff(x, prefix, prob=None, N_bins=100, histrange=None, x_lim=None,
                y_lim=None, x_label=None, y_label=None, legend=None):
    """
    this function takes in parameter arrays
    bins the data then calculates the percentage difference
    actually forgot where I used this
    """
    #fig = pylab.figure()
    # find out size of array
    totalsize = len(x)
    print 'data size of each variable is ', totalsize
    # divide total number of data points into nparts-1
    nparts = 101
    d = (nparts - 1, 5)
    reduced_d = (nparts-2, 5)
    data = numpy.zeros(d)
    x_perdiff = numpy.zeros(reduced_d)

    # iterate from 1 to nparts-1
    for n in range(1, nparts):
            # print i,"th iteration"
        # size of each of the n parts:
        partsize = totalsize*(n)/(nparts-1)
        hist, binedges, tmp = \
            astroMLhist(x[:partsize], bins="knuth",
                 histtype='step', weights=prob[:partsize],
                 range=histrange, color='k', linewidth=2)

        # Calculate the location and confidence intervals
        # Since my location and confidence calculations can't take weighted data I
        # need to use the weighted histogram data in the calculations
        for i in numpy.arange(N_bins):
            if i == 0:
                x_binned = \
                    numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
            else:
                x_temp = \
                    numpy.ones(hist[i]) * (binedges[i] + binedges[i + 1]) / 2
                x_binned = numpy.concatenate((x_binned, x_temp))
        # print 'len of x_binned is ',len(x_binned)
        loc = biweightLoc(x_binned)
        ll_68, ul_68 = bcpcl(loc, x_binned, 1)
        ll_95, ul_95 = bcpcl(loc, x_binned, 2)
        # this will store the data
        data[n-1] = [loc, ll_68, ul_68, ll_95, ul_95]

    filename = prefix+'_histplot1D_percentdiff.png'

    pylab.axvline(loc, ls='--', linewidth=2, label='$C_{BI}$', color="k")
    pylab.axvline(ll_68, ls='-.', linewidth=2, color='#800000',
                  label='68% $IC_{B_{BI}}$')
    pylab.axvline(ul_68, ls='-.', linewidth=2, color='#800000')
    pylab.axvline(ll_95, ls=':', linewidth=2, color='#0000A0',
                  label='95% $IC_{B_{BI}}$')
    pylab.axvline(ul_95, ls=':', linewidth=2, color='#0000A0')

    pylab.savefig(filename, dpi=300, bbox_inches='tight')
    pylab.close()

    print '\n'+prefix + ' data is '
    print data
    print '      '

    for n in range(1, nparts-1):
        x_perdiff[n-1] = (data[nparts - 2]-data[n - 1])*2 * \
            100/(data[nparts-2]+data[n-1])
    print prefix+' per diff is '
    print x_perdiff
    print '      '

    # this will invert the array, now disabled
    #x_perdiff = x_perdiff[::-1]

    return x_perdiff


def boxplot(result, y, tick_label, key, size=16, save=False):
    """ plot my version of boxplot
    usage: apply to each row of a dataframe with the results from
    histplot1d

    Parameters
    ------------
    result = list of numbers of size 5
        denoting [location, CI68_lower, CI68_upper, CI95_lower, CI95_upper]
    y = vertical location
    tick_label = list of numbers
        to be used as y_ticks
    key = string
        denotes the name of the variable on the x axis
    size = integer, font size of labels

    example usage:
    -------------
    tick_label = np.arange(alpha_lower - alpha_interval,
                    alpha_upper + alpha_interval, alpha_interval)
    for key in par:
        for alpha in alpha0:
            # results is a dataframe with index = [[str(alpha)], [key]]
            boxplot(results.ix[str(alpha), key], alpha, tick_label, key)
            plt.savefig("CI_" + key + '.pdf', bbox_inches='tight')
        plt.close()

    """
    import matplotlib.pyplot as plt
    CI68 = np.arange(result[1], result[2], 0.01)
    CI95 = np.arange(result[3], result[4], 0.01)
    plt.plot(CI95, y * np.ones(CI95.size), 'g-', label='95% CI')
    plt.plot(CI68, y * np.ones(CI68.size), 'b-')
    plt.plot(CI68[0], y, 'b|', markeredgewidth=2, label='68% CI')
    plt.plot(CI68[-1], y, 'b|', markeredgewidth=2)
    plt.plot(result[0], y, 'r|', markeredgewidth=2, label='location')
    plt.yticks(tick_label)
    plt.ylabel(r'$\alpha_0$', size=size)
    plt.xlabel(key, size=size)
    plt.show()

    return


def N_by_M_plot_contour(data, Nvar_list, Mvar_list, space, axlims=None,
                        Nbins_2D=None, axlabels=None, xlabel_to_rot=None,
                        histran=None, figsize=6, fontsize=11):
    """create a N by M matrix of 2D contour plots
    data = dataframe that contain the data of all the variables to be plots
    Nvar_list = list of strings - denotes the column header names
        that needs to be plotted on x-axes, col names correspond to xlabel
    Mvar_list = list of strings - denotes the column header names
        that needs to be plotted on y-axes, col names correspond to ylabel
    space = float, px of space that is added between subplots
    axlims = dictionary, keys are the strings in var_list,
        each value is a tuple of (low_lim, up_lim) to denote the limit
        of values to be plotted
    Nbins_2D = dictionary, keys are in format of tuples of
        (x_col_str, y_col_str) to denote which subplot you are referring to
    axlabels = dictionary, keys correspond to the variable names, values
        strings that correspond to the labels to be put on corresponding
        axes
    xlabel_to_rot = dictionary,
        key is the the key for the labels to be rotated,
        value is the degree to be rotated
    histran = dictionary,
        some keys has to be the ones for the plots, value are in
        form of (lowerhist_range, upperhist_range)
    figsize = integer, figuares are squared this refers to the side length
    fontsize = integer, denotes font size of the labels

    Stability: Not entirely tested, use at own risk

    Author: Karen Ng
    """
    from matplotlib.ticker import MaxNLocator

    N = len(Nvar_list)
    M = len(Mvar_list)

    print 'creating input-input figure with dimension ' + \
        '{0} rows by {1} cols'.format(M, N)

    # begin checking if inputs make sense
    assert N <= len(axlabels), "length of axlabels is wrong"
    assert M <= len(axlabels), "length of axlabels is wrong"

    compare_Nvar = np.sum([Nvar in data.columns for Nvar in Nvar_list])
    assert compare_Nvar == len(Nvar_list), "variable to be plotted not in df"

    compare_Mvar = np.sum([Mvar in data.columns for Mvar in Mvar_list])
    assert compare_Mvar == len(Mvar_list), "variable to be plotted not in df"

    keys = comb_zip(Nvar_list, Mvar_list)
    if Nbins_2D is not None:
        compare_keys = np.sum([key in Nbins_2D.keys() for key in keys])
        assert compare_keys == len(Nbins_2D), "Nbins_2D key error"
    else:
        Nbins_2D = {key: 50 for key in keys}

    if axlims is None:
        axlims = {key: (None, None) for key in Nvar_list + Mvar_list}

    # impossible for the matrix plot not to be squared in terms of dimensions
    # set each of the subplot to be squared with the figsize option
    f, axarr = plt.subplots(M, N, figsize=(figsize * N / M, figsize))
    f.subplots_adjust(wspace=space, hspace=space)

    # remove unwanted row axes tick labels
    plt.setp([a.get_xticklabels() for i in range(M - 1)
              for a in axarr[i, :]], visible=False)

    ## remove unwanted column axes tick labels
    plt.setp([a.get_yticklabels() for i in range(1, N)
              for a in axarr[:, i]], visible=False)

    ## rotate the xlabels appropriately
    if xlabel_to_rot is not None:
        match_ix = [Nvar_list.index(item) for item in xlabel_to_rot.keys()]
        # ok to use for-loops for small number of iterations
        for ix in match_ix:
            labels = axarr[M - 1, ix].get_xticklabels()
            for label in labels:
                label.set_rotation(xlabel_to_rot[Nvar_list[ix]])

    ## create axes labels
    if axlabels is not None:
        for j in range(M):
            axarr[j, 0].set_ylabel(axlabels[Mvar_list[j]], fontsize=fontsize)
        for i in range(N):
            axarr[M - 1, i].set_xlabel(axlabels[Nvar_list[i]],
                                       fontsize=fontsize)

    # fix the number of bins on the axis by nbins
    ## avoid overlapping lowest and highest ticks mark with prune
    # option pruning both upper and lower bound
    for m in range(M):
        ax1 = axarr[m, 0]
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
    for n in range(N):
        ax2 = axarr[M - 1, n]
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    ## start plotting the 2D contours
    for i in range(M):
        for j in range(N):
            #print "axarr[i, j] has indices {0}".format((i, j))
            #print "x axis label = {0}".format(Nvar_list[j])
            #print "y axis label = {0}".format(Mvar_list[i])
            histplot2d_part(axarr[i, j],
                            data[Nvar_list[j]],
                            data[Mvar_list[i]],
                            prob=data['prob'],
                            N_bins=Nbins_2D[(Nvar_list[j],
                                             Mvar_list[i])],
                            x_lim=axlims[Nvar_list[j]],
                            y_lim=axlims[Mvar_list[i]])

    return


def N_by_N_lower_triangle_plot(data, space, var_list, axlims=None,
                               Nbins_2D=None, axlabels=None,
                               xlabel_to_rot=None, histran=None, figsize=6,
                               fontsize=12):
    """ create a N by N matrix of plots
    with the top plot of each row showing a density plot in 1D
    and the remaining plots being 2D contour plots
    df = dataframe that contain the data of all the variables to be plots
    space = float, px of space that is added between subplots
    var_list = list of strings - denotes the column header names
        that needs to be plotted
    axlims = dictionary, keys are the strings in var_list,
        each value is a tuple of (low_lim, up_lim) to denote the limit
        of values to be plotted
    Nbins_2D = dictionary, keys are in format of tuples of
        (x_col_str, y_col_str) to denote which subplot you are referring to
    axlabels = dictionary, keys correspond to the variable names
    xlabel_to_rot = dictionary,
        key is the the key for the labels to be rotated,
        value is the degree to be rotated
    histran = dictionary,
        some keys has to be the ones for the plots, value are in
        form of (lowerhist_range, upperhist_range)
    figsize = integer, figuares are squared this refers to the side length
    fontsize = integer, denotes font size of the labels

    Stability: Not entirely tested, use at own risk
    """
    from matplotlib.ticker import MaxNLocator

    # begin checking if inputs make sense
    N = len(var_list)
    assert N <= len(axlabels), "length of axlabels is wrong"
    assert N >= 2, "lower triangular contour plots require more than 2\
        variables in the data"

    for var in var_list:
        assert var in data.columns, "variable to be plotted not in df"

    if axlabels is None:
        axlabels = {key: key for key in var_list}

    if xlabel_to_rot is None:
        xlabel_to_rot = {key: 0 for key in var_list}

    if histran is None:
        histran = {key: None for key in var_list}

    if axlims is None:
        axlims = {key: (None, None) for key in var_list}


    if Nbins_2D is None:
        keys = comb_zip(Nvar_list, Nvar_list)
        Nbins_2D = {key : 50 for key in keys}

    # impossible for the matrix plot not to be squared in terms of dimensions
    # set each of the subplot to be squared with the figsize option
    f, axarr = pylab.subplots(N, N, figsize=(figsize, figsize))
    f.subplots_adjust(wspace=space, hspace=space)

    # remove unwanted plots on the upper right
    plt.setp([a.get_axes() for i in range(N - 1)
              for a in axarr[i, i + 1:]], visible=False)

    # remove unwanted row axes tick labels
    plt.setp([a.get_xticklabels() for i in range(N - 1)
              for a in axarr[i, :]], visible=False)

    # remove unwanted column axes tick labels
    plt.setp([axarr[0, 0].get_yticklabels()], visible=False)
    plt.setp([a.get_yticklabels() for i in range(N - 1)
              for a in axarr[i + 1, 1:]], visible=False)

    # create axes labels
    if axlabels is not None:
        for j in range(1, N):
            axarr[j, 0].set_ylabel(axlabels[var_list[j]], fontsize=fontsize)
        for i in range(N):
            axarr[N - 1, i].set_xlabel(axlabels[var_list[i]],
                                       fontsize=fontsize)

    # avoid overlapping lowest and highest ticks mark
    for n in range(N):
        ax1 = axarr[n, 0]
        ax1.yaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))
        ax2 = axarr[N - 1, n]
        ax2.xaxis.set_major_locator(MaxNLocator(nbins=6, prune="both"))

    # rotate the xlabels appropriately
    if xlabel_to_rot is not None:
        match_ix = [var_list.index(item) for item in xlabel_to_rot.keys()]
        # ok to use for-loops for small number of iterations
        for ix in match_ix:
            labels = axarr[N - 1, ix].get_xticklabels()
            for label in labels:
                label.set_rotation(xlabel_to_rot[var_list[ix]])

    # start plotting the diagonal
    for i in range(N):
        histplot1d_part(axarr[i, i], np.array(data[var_list[i]]), data['prob'],
                        N_bins="knuth", histrange=histran[var_list[i]],
                        x_lim=axlims[var_list[i]])

    # start plotting the lower triangle when row no > col no
    for i in range(N):
        for j in range(i):
            histplot2d_part(axarr[i, j], data[var_list[j]],
                            data[var_list[i]],
                            prob=data['prob'],
                            N_bins=Nbins_2D[(var_list[j], var_list[i])],
                            x_lim=axlims[var_list[j]],
                            y_lim=axlims[var_list[i]])

    return


# ------ newly added functions for comparing before and after adding prior--
def plot_perdiff(perdiff, labels, title):
    """
    Plot the percentage differences between different % of data within
    Will's   2 million iterations
    """
    fig = plt.figure()
    for i in range(5):
        plt.plot(range(1, len(perdiff[:, 1]) + 1),
                 perdiff[:, i], label=labels[i])
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc=4, bbox_to_anchor=(1.375, .5))
    minorLocator = MultipleLocator(2)

    plt.title(title)
    plt.grid()
    plt.grid(True, which='minor')
    plt.xlabel('% of iterations')
    plt.ylabel('Percent difference')
    ax.xaxis.set_minor_locator(minorLocator)
    plt.savefig(title+'.png', dpi=300, bbox_inches='tight')
    return fig


def prior_diff(data1, data2, prefix, prob1=None, prob2=None, N_bins=100,
               histrange=None, x_lim=None, y_lim=None, x_label=None,
               y_label=None, legend=None):
    """
    Plot the histograms between the same set of data with
    and without applying prior on the same plot
    """

    fig = pylab.figure()

    # bin data 1 first
    hist2, binedges2, tmp2 = \
        pylab.hist(data2, bins=N_bins, histtype='step',
                   weights=prob2, range=histrange, color='#ff0000', linewidth=2)
    hist1, binedges1, tmp1 = \
        pylab.hist(data1, bins=N_bins, histtype='step',
                   weights=prob1, range=histrange, color='#0000ff', linewidth=2)

    # Calculate the location and %confidence intervals for data 1
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned1 = numpy.ones(hist1[i])*(binedges1[i]+binedges1[i+1])/2
        else:
            x_temp1 = numpy.ones(hist1[i])*(binedges1[i]+binedges1[i+1])/2
            x_binned1 = numpy.concatenate((x_binned1, x_temp1))
    loc1 = biweightLoc(x_binned1)
    ll_68_1, ul_68_1 = bcpcl(loc1, x_binned1, 1)
    ll_95_1, ul_95_1 = bcpcl(loc1, x_binned1, 2)

    # Create location and confidence interval line plots
    # this should totally be replaced
    pylab.plot((loc1, loc1), (pylab.ylim()[0], pylab.ylim()[1]),
               '--', linewidth=2, color='#6495ed', label='$C_{BI}$')
    pylab.plot((ll_68_1, ll_68_1), (pylab.ylim()[0], pylab.ylim()[1]),
               '-.', linewidth=2, color='#7fffd4', label='68% $IC_{B_{BI}}$')
    pylab.plot((ul_68_1, ul_68_1),
               (pylab.ylim()[0], pylab.ylim()[1]), '-.', linewidth=2, color='#7fffd4')
    pylab.plot((ll_95_1, ll_95_1), (pylab.ylim()[0], pylab.ylim()[1]),
               ':', linewidth=2, color='#87ceeb', label='95% $IC_{B_{BI}}$')
    pylab.plot((ul_95_1, ul_95_1),
               (pylab.ylim()[0], pylab.ylim()[1]), ':', linewidth=2, color='#87ceeb')

    # Calculate the location and %confidence intervals for data 2
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned2 = numpy.ones(hist2[i])*(binedges2[i]+binedges2[i+1])/2
        else:
            x_temp2 = numpy.ones(hist2[i])*(binedges2[i]+binedges2[i+1])/2
            x_binned2 = numpy.concatenate((x_binned2, x_temp2))
    loc2 = biweightLoc(x_binned2)
    ll_68_2, ul_68_2 = bcpcl(loc2, x_binned2, 1)
    ll_95_2, ul_95_2 = bcpcl(loc2, x_binned2, 2)

    # Create location and confidence interval line plots
    pylab.plot((loc2, loc2), (pylab.ylim()[0], pylab.ylim()[1]),
               '--', linewidth=2, color='#ff4500', label='$C_{BI}$')
    pylab.plot((ll_68_2, ll_68_2), (pylab.ylim()[0], pylab.ylim()[1]),
               '-.', linewidth=2, color='#ff8c00', label='68% $IC_{B_{BI}}$')
    pylab.plot((ul_68_2, ul_68_2),
               (pylab.ylim()[0], pylab.ylim()[1]), '-.', linewidth=2, color='#ff8c00')
    pylab.plot((ll_95_2, ll_95_2), (pylab.ylim()[0], pylab.ylim()[1]),
               ':', linewidth=2, color='#ffa500', label='95% $IC_{B_{BI}}$')
    pylab.plot((ul_95_2, ul_95_2),
               (pylab.ylim()[0], pylab.ylim()[1]), ':', linewidth=2, color='#ffa500')

    # create labels for the plots
    if x_label != None:
        pylab.xlabel(x_label, fontsize=20)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=20)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        pylab.legend()
    # fontsize=14
    #ax = pylab.gca()
    # for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)

    filename = prefix+'_prior_diff'
    pylab.savefig(title+'.png', dpi=300, bbox_inches='tight')

    print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(
        prefix, loc1, ll_68_1, ul_68_1, ll_95_1, ul_95_1)

    print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(
        prefix, loc2, ll_68_2, ul_68_2, ll_95_2, ul_95_2)

    return loc2, ll_68_2, ul_68_2, ll_95_2, ul_95_2


def prior_diff_pdf(data1, data2, prefix, prob1=None, prob2=None, N_bins=100,
                   histrange=None, x_lim=None, y_lim=None, x_label=None,
                   y_label=None, legend=None):
    """
    Plot the pdf between the same set of data with
    and without applying prior on the same plot
    """

    # bin data 2 first
    hist2, binedges2, tmp2 = \
        pylab.hist(data2, bins=N_bins, histtype='step', weights=prob2,
                   range=histrange, color='#ff0000', linewidth=2)
    # bin data 1
    hist1, binedges1, tmp1 = \
        pylab.hist(data1, bins=N_bins, histtype='step', weights=prob1,
                   range=histrange, color='#0000ff', linewidth=2)
    pylab.close()

    fig = pylab.figure()
    # plot the pdf for the data
    pdf2, histbin2, tmp_pdf2 = \
        pylab.hist(data2, bins=N_bins, normed=1,
                   histtype='step', weights=prob2, range=histrange,
                   color='#ff0000', linewidth=2)

    pdf1, histbin1, tmp_pdf1 = pylab.hist(data1, bins=N_bins, normed=1,
                                          histtype='step', weights=prob1, range=histrange, color='#0000ff',
                                          linewidth=2)

    # Calculate the location and %confidence intervals for data 1
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned1 = numpy.ones(histbin1[i]) *\
                (binedges1[i]+binedges1[i+1])/2
        else:
            x_temp1 = numpy.ones(hist1[i])*(binedges1[i]+binedges1[i+1])/2
            x_binned1 = numpy.concatenate((x_binned1, x_temp1))
    loc1 = biweightLoc(x_binned1)
    ll_68_1, ul_68_1 = bcpcl(loc1, x_binned1, 1)
    ll_95_1, ul_95_1 = bcpcl(loc1, x_binned1, 2)

    # adjust the max ylim so it does not look weird
    ylim_max = pdf2.max()*1.2

    # Create location and confidence interval line plots
    pylab.plot((loc1, loc1), (pylab.ylim()[0], ylim_max), '--', linewidth=2,
               color='#6495ed', label='$C_{BI}$')
    pylab.plot((ll_68_1, ll_68_1), (pylab.ylim()[0], ylim_max), '-.',
               linewidth=2, color='#7fffd4', label='68% $IC_{B_{BI}}$')
    pylab.plot((ul_68_1, ul_68_1), (pylab.ylim()[0], ylim_max), '-.',
               linewidth=2, color='#7fffd4')
    pylab.plot((ll_95_1, ll_95_1), (pylab.ylim()[0], ylim_max), ':',
               linewidth=2, color='#87ceeb', label='95% $IC_{B_{BI}}$')
    pylab.plot((ul_95_1, ul_95_1), (pylab.ylim()[0], ylim_max), ':',
               linewidth=2, color='#87ceeb')

    # Calculate the location and %confidence intervals for data 2
    # Since my location and confidence calculations can't take weighted data I
    # need to use the weighted histogram data in the calculations
    for i in numpy.arange(N_bins):
        if i == 0:
            x_binned2 = numpy.ones(hist2[i])*(binedges2[i]+binedges2[i+1])/2
        else:
            x_temp2 = numpy.ones(hist2[i])*(binedges2[i]+binedges2[i+1])/2
            x_binned2 = numpy.concatenate((x_binned2, x_temp2))
    loc2 = biweightLoc(x_binned2)
    ll_68_2, ul_68_2 = bcpcl(loc2, x_binned2, 1)
    ll_95_2, ul_95_2 = bcpcl(loc2, x_binned2, 2)

    # Create location and confidence interval line plots
    # if y_lim == None:
    # else:
    #    ylim_max = y_lim[1]
    #    ylim_min = y_lim[0]
    # if x_lim == None:
    #    xlim_max = pylab.xlim()[1]
    #    xlim_min = pylab.xlim()[0]
    # else:
    #    xlim_max = x_lim[1]
    #    xlim_min = x_lim[0]
    pylab.plot((loc2, loc2), (pylab.ylim()[0], ylim_max), '--', linewidth=2,
               color='#ff4500', label='$C_{BI}$')
    pylab.plot((ll_68_2, ll_68_2), (pylab.ylim()[0], ylim_max), '-.',
               linewidth=2, color='#ff8c00', label='68% $IC_{B_{BI}}$')
    pylab.plot((ul_68_2, ul_68_2), (pylab.ylim()[0], ylim_max), '-.',
               linewidth=2, color='#ff8c00')
    pylab.plot((ll_95_2, ll_95_2), (pylab.ylim()[0], ylim_max), ':',
               linewidth=2, color='#ffa500', label='95% $IC_{B_{BI}}$')
    pylab.plot((ul_95_2, ul_95_2), (pylab.ylim()[0], ylim_max), ':',
               linewidth=2, color='#ffa500')

    # create labels for the plots
    if x_label != None:
        pylab.xlabel(x_label, fontsize=20)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=20)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
        pylab.legend()
    # fontsize=14
    #ax = pylab.gca()
    # for tick in ax.xaxis.get_major_ticks():
        # tick.label1.set_fontsize(fontsize)
    pylab.ylim(0, pdf2.max()*1.2)

    filename = prefix+'_prior_diff'
    pylab.savefig(filename, dpi=300, bbox_inches='tight')

    print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(
        prefix, loc1, ll_68_1, ul_68_1, ll_95_1, ul_95_1)

    print '{0}, {1:0.4f}, {2:0.4f}, {3:0.4f}, {4:0.4f}, {5:0.4f}'.format(
        prefix, loc2, ll_68_2, ul_68_2, ll_95_2, ul_95_2)

    return loc2, ll_68_2, ul_68_2, ll_95_2, ul_95_2


def histplot2d_2contour(
    x1, y1, x, y, prefix, prob1=None, prob=None, N_bins=100,
    histrange=None, x_lim=None, y_lim=None, x_label=None, y_label=None,
    legend=None):
    """
    My function for plotting 2 sets of 2d contour
    Create the confidence interval plot
    """
    if histrange is None:
        if prob1 is not None:
            H1, xedges1, yedges1 = \
                numpy.histogram2d(x1, y1, bins=N_bins, weights=prob1)
        elif prob1 is None:
            H1, xedges1, yedges1 = \
                numpy.histogram2d(x1, y1, bins=N_bins)
    else:
        if prob1 is not None:
            H1, xedges1, yedges1 = \
                numpy.histogram2d(x1, y1, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                         [histrange[2], histrange[3]]],
                                  weights=prob1)
        elif prob is None:
            H1, xedges1, yedges1 = \
                numpy.histogram2d(x1, y1, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                         [histrange[2], histrange[3]]])
    H1 = numpy.transpose(H1)
    # Flatten H
    h = numpy.reshape(H1, (N_bins ** 2))
    # Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum / h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum / h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]

    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x1 = xedges1[:-1]+(xedges1[1]-xedges1[0])/2
    y1 = yedges1[:-1]+(yedges1[1]-yedges1[0])/2
    X1, Y1 = numpy.meshgrid(x1, y1)

    fig = pylab.figure()
    # Countours
    CS = pylab.contour(X1, Y1, H1, (h_2sigma, h_1sigma),
                       linewidths=(2, 2), colors=('#a4a4a4', '#6e6e6e'))
    # imshow
    #im = pylab.imshow(H1,cmap=pylab.cm.gray)
    # pylab.pcolor(X1,Y1,H1,cmap=pylab.cm.white)

    if x_label != None:
        pylab.xlabel(x_label, fontsize=14)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=14)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    # if legend != None:
    # Dummy lines for legend
        # 800000 is for light gray - 68% 1 sigma
        # 0000A0 is for whitesmoke - 95% confidence 3 sigma
    # pylab.plot((0,1),(0,1),c='#f5fffa',linewidth=2,label=('68%'))
    # pylab.plot((0,1),(0,1),c='#d3d3d3',linewidth=2,label=('95%'))
        # pylab.legend(scatterpoints=1)
    fontsize = 20
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    # SECOND contour
    # Create the confidence interval plot for the second sets of contour
    if histrange == None:
        if prob != None:
            H, xedges, yedges = numpy.histogram2d(
                x, y, bins=N_bins, weights=prob)
        elif prob == None:
            H, xedges, yedges = numpy.histogram2d(x, y, bins=N_bins)
    else:
        if prob != None:
            H, xedges, yedges = \
                numpy.histogram2d(x, y, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                         [histrange[2], histrange[3]]],
                                  weights=prob)
        elif prob == None:
            H, xedges, yedges = \
                numpy.histogram2d(x, y, bins=N_bins,
                                  range=[[histrange[0], histrange[1]],
                                         [histrange[2], histrange[3]]])
    H = numpy.transpose(H)
    # Flatten H
    h = numpy.reshape(H, (N_bins ** 2))
    # Sort h from smallest to largest
    index = numpy.argsort(h)
    h = h[index]
    h_sum = numpy.sum(h)
    # Find the 2 and 1 sigma levels of the MC hist
    for j in numpy.arange(numpy.size(h)):
        if j == 0:
            runsum = h[j]
        else:
            runsum += h[j]
        if runsum/h_sum <= 0.05:
            # then store the value of N at the 2sigma level
            h_2sigma = h[j]
        if runsum/h_sum <= 0.32:
            # then store the value of N at the 1sigma level
            h_1sigma = h[j]
    # Create the contour plot using the 2Dhist info
    # define pixel values to be at the center of the bins
    x = xedges[:-1]+(xedges[1]-xedges[0])/2
    y = yedges[:-1]+(yedges[1]-yedges[0])/2
    X, Y = numpy.meshgrid(x, y)

    # Coutours
    CS = pylab.contour(X, Y, H, (h_2sigma, h_1sigma), linewidths=(2, 2))
    # imshow
    #im = pylab.imshow(H,cmap=pylab.cm.Blues)
    pylab.pcolor(X, Y, H, cmap=pylab.cm.gray_r)

    if x_label != None:
        pylab.xlabel(x_label, fontsize=20)
    if y_label != None:
        pylab.ylabel(y_label, fontsize=20)
    if x_lim != None:
        pylab.xlim(x_lim)
    if y_lim != None:
        pylab.ylim(y_lim)
    if legend != None:
    # Dummy lines for legend
        # 800000 is for Maroon color - 68% 1 sigma
        # 0000A0 is for blue color - 95% confidence 3 sigma
        pylab.plot((0, 1), (0, 1), c='#87cefa', linewidth=2, label=('68%'))
        pylab.plot((0, 1), (0, 1), c='#6495ed', linewidth=2, label=('95%'))
        pylab.legend(scatterpoints=1)
    fontsize = 20
    ax = pylab.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)

    filename = prefix+'_histplot2d_combine2'
    pylab.savefig(filename, dpi=300, bbox_inches='tight')

    return fig


def plot_perdiff(perdiff, labels, title):
    """
    Plot the percentage differences between different % of data within
    Will's   2 million iterations
    """
    fig = plt.figure()
    for i in range(5):
        plt.plot(range(1, len(perdiff[:, 1])+1),
                 perdiff[:, i], label=labels[i])
    ax = plt.subplot(111)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc=4, bbox_to_anchor=(1.375, .5))
    minorLocator = MultipleLocator(2)

    plt.title(title)
    plt.grid()
    plt.grid(True, which='minor')
    plt.xlabel('% of iterations')
    plt.ylabel('Percent difference')
    ax.xaxis.set_minor_locator(minorLocator)
    plt.savefig(title+'.png', dpi=300, bbox_inches='tight')
    return fig


# Create the prior from the radio relic constraints
# we know that if the two subclusters are within 0.5 Mpc to 1.5 Mpc
# then the detection of a radio relic is possible
# d_3d is in Mpc
# return a mask that can be applied to other data arrays
# also return how many non-zero entries
# def radio_dist_prior(d_3D, mask, d_3Dmax = 3.0, d_3Dmin = 1.0):
#    '''
#    to be modified such that this takes in the range for the uniform prior
#    input:
#    d_3D = numpy array to be masked
#    radiomask = numpy array that is contains either 1 or 0
#    d_3Dmax = float, the upper limit to be masked out
#    d_3Dmin = float, the lower limit to be masked out
#    '''
#    count = 0
#    for r in range(len(d_3D)):
#        if (d_3D[r]>d_3Dmax)  or (d_3D[r]< d_3Dmin):
#            radiomask[r] = 0
#            count += 1
#    count = len(radiomask) - count
#    return radiomask, count

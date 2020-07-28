import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import matplotlib as mpl

from tqdm.autonotebook import tqdm
from IPython.display import HTML

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from shapely.geometry import Polygon
from COSIpy import FISBEL
from COSIpy import dataset
from COSIpy import GreatCircle
from COSIpy import angular_distance

import pickle
import pystan

class fit():
    """
    Fitting class that includes the dataset to be analysed, the pointings, the response, and a background model.
    :option: bg_only:   Default = False: performs a background only fit when True
    :option: priors:    Default: uninformative (normal) priors for the sky;
    :option: verbose:   Default = False: more verbose output
    """

    
    def __init__(self,
                 dataset,          # COSIpy dataset
                 pointings,        # POINTINGS object
                 response,         # SkyResponse object for certain source position
                 background,       # BG object including cuts and tracer
                 bg_only=False,    # True performs BG only fit
                 priors=None,      # can set priors for sky components
                 verbose=False):   # more verbose output

        # init objects
        self.dataset = dataset
        self.pointings = pointings
        self.response = response
        self.background = background

        # construct data set with reduced CDS (ignore bins that are always zero)
        self.count_data = self.reduce_dataset_CDS()
        
        self.bg_only = bg_only
        self.priors = priors
        self.verbose = verbose

        # make dictionaries for fit with Stan
        # load corresponding Stan model
        if not self.bg_only:
            self.data_per_energy_bin = self.make_dictionaries_for_stan()
            self.load_stan_model()
        else:
            if self.verbose:
                print('This will be a background only fit.')
            self.data_per_energy_bin = self.make_dictionaries_for_stan_bg_only()
            self.load_stan_model_bg_only()
    
    
    def reduce_dataset_CDS(self):
    """
    Reduce data space, given background response, to ignore bins that are always zero
    """
    
        # init list of data sets per bin
        # will be irregularly-shaped because CDS population depends on energy
        count_data_reduced = []

        # loop ove renergies
        for i in range(self.dataset.energies.n_energy_bins):
            
            # reshape count data to reduce CDS if possible
            # this combines the 3 CDS angles into a 1D array for all times at the chosen energy
            yp_tmp = self.dataset.binned_data[:,i,:,:].reshape(self.dataset.times.n_time_bins,
                                                          self.background.bg_model.shape[2]*self.background.bg_model.shape[3])
    
            # reshape count data grid the same way as backgrorund and choose only non-zero indices
            yp_tmp = yp_tmp[:,self.background.calc_this[i]]

            # append reduced data set
            count_data_reduced.append(yp_tmp)
            
        return count_data_reduced
    
    
    def make_dictionaries_for_stan(self):
    """
    Create dictionaries that can be read in by the Stan model to fit the data
    """

        # init dictionaries for each energy bini
        all_dicts = []

        # loop over energy bins
        for i in range(self.dataset.energies.n_energy_bins):
            Np, Nrsp = self.background.bg_model_reduced[i].shape         # initialise sizes of arrays
            N = Np*Nrsp                                                  # total number of data points

            # right now, only one sky model allowed to be fitted
            Nsky = 1

            # standard priors scaling the initial flux as in response calculation
            if self.priors == None:

                # mean
                mu_flux_scl = np.array([1.])    # prior centroids for sky, we use 10 because we are ignorant;
                # this has to be an array because it could be more than one
                # std
                sigma_flux_scl = np.array([1.]) # same for the width (so, easily 0 but also high values possible)
            
            else:

                # set priors yourself
                mu_flux_scl = np.array([self.priors[0]])
                sigma_flux_scl = np.array([self.priors[1]])

            # priors for backgrorund model components
            # initially normalised to 1, so mean would be 1, variance very large (uninformative)
            mu_Abg = 1.       # for the moment set to a useful value if bg model is ~normalised to data
            sigma_Abg = 1e4   # same

            # dictionary for data set and prior
            data2D = dict(N = Nrsp,                                                      # number of CDS bins
                          Np = Np,                                                       # number of observations
                          Nsky = Nsky,                                                   # number of sky models (now: 1)
                          Ncuts = self.background.Ncuts,                                 # number of background cuts / renormalisations
                          bg_cuts = self.background.bg_cuts,                             # bg cuts at
                          bg_idx_arr = self.background.idx_arr,                          # bg cut indices
                          y = self.count_data[i].ravel().astype(int),                    # data
                          bg_model = self.background.bg_model_reduced[i],                # background model 
                          conv_sky = self.response.sky_response[i].reshape(Nsky,Np,Nrsp),# this has to be reshaped because it could be more than one
                          mu_flux = mu_flux_scl,                                         # priors for sky (mean)
                          sigma_flux = sigma_flux_scl,                                   # std
                          mu_Abg = mu_Abg,                                               # BG mean
                          sigma_Abg = sigma_Abg)                                         # std

            # append dictionary for energy bin
            all_dicts.append(data2D)
            
        return all_dicts
    
    
    def make_dictionaries_for_stan_bg_only(self):
    """
    Same as above just for background-only fit
    """
        # init dictionary per energy bin
        all_dicts = []

        # loop over energies
        for i in range(self.dataset.energies.n_energy_bins):
            
            Np, Nrsp = self.background.bg_model_reduced[i].shape         # initialise sizes of arrays
            N = Np*Nrsp                                                  # total number of data points

            mu_Abg = 1.                     # for the moment set to a useful value if bg model is ~normalised to data
            sigma_Abg = 100.                # same

            # dictionary for data set and prior
            data2D = dict(N = Nrsp,
                          Np = Np,
                          Ncuts = self.background.Ncuts,
                          bg_cuts = self.background.bg_cuts,
                          bg_idx_arr = self.background.idx_arr,
                          y = self.count_data[i].ravel().astype(int),
                          bg_model = self.background.bg_model_reduced[i],
                          mu_Abg = mu_Abg,
                          sigma_Abg = sigma_Abg)
            all_dicts.append(data2D)
            
        return all_dicts
    
    
    def load_stan_model(self):
    """
    Loading the Stan model COSImodfit.stan.
    Compiles it if not already done.
    """
        try:
            #read COSImodefit.pkl (if already compiled)
            self.model = pickle.load(open('COSImodfit.pkl', 'rb'))
            
        except:
            print('Model not yet compiled, doing that now (might take a while).')
            ## compile model (if not yet compiled):
            self.model = pystan.StanModel('COSImodfit.stan')

            ## save it to the file 'filename.pkl' for later use
            with open('COSImodfit.pkl', 'wb') as f:
                pickle.dump(self.model, f)
    
    
    def load_stan_model_bg_only(self):
    """
    Loading Stan model for background only.
    Compiles it of not already done.
    """
        try:
            #read COSImodfit_BGonly.pkl (if already compiled)
            self.model = pickle.load(open('COSImodfit_BGonly.pkl', 'rb'))
            
        except:
            print('Model not yet compiled, doing that now (might take a while).')
            ## compile model (if not yet compiled):
            self.model = pystan.StanModel('COSImodfit_BGonly.stan')

            ## save it to the file 'filename.pkl' for later use
            with open('COSImodfit_BGonly.pkl', 'wb') as f:
                pickle.dump(self.model, f)
                
    
    def MAP_solution(self,guess=1.0,method='LBFGS'):
    """
    Performs optimisation of the joint posterior distribution and returns Median A-Posteriori (MAP) point estimate.
    Returns no error bars and serves as quick cross check or for calls to likelihood ratio tests.
    Creates array .diff_flux_map that includes the differential flux in units of ph/cm2/s/keV for each energy bin.
    Saves all fit results and quality in .fit_pars_map.
    """
        # init arrays to save info
        self.fit_pars_map = []
        self.diff_flux_map = np.zeros(self.dataset.energies.n_energy_bins)
        
        # loop over energy bins
        for i in tqdm(range(self.dataset.energies.n_energy_bins),desc='Loop over energy bins:'):
            if self.verbose:
                print('Start optimising energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+'...')
                
            # not necessary in general, but good for quicklook MAP estimate
            init = {}
            if not self.bg_only:
                init['flux'] = np.array([guess])
                init['Abg'] = np.repeat(1.0,self.background.Ncuts)
            else:
                init['Abg'] = np.repeat(1.5,self.background.Ncuts)
                
            # optimiising the model
            op = self.model.optimizing(data=self.data_per_energy_bin[i],
                                       verbose=False,init=init,as_vector=False,algorithm=method)#,tol_rel_grad=1e5)

            # append the result
            self.fit_pars_map.append(op)

            # calculate the flux
            if not self.bg_only:
                self.diff_flux_map[i] = self.fit_pars_map[i]['par']['flux']/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm
            
            
    def fit(self,iters=1000,pars=['flux','Abg']):
        """
        Fitting COSIpy fit object of a data set with pointing definition, background model, and (now only) point source response.
        Fitting background only is only possible when object is initialised with bg_only = True.
        :option: par    Parameters to save in fit object: pars=['flux','Abg','model_tot','model_bg','model_sky','ppc'], default pars=['flux','Abg'],
                        i.e. no models will be save, only the fitted parameters.
        :option: iters  Number of iterations for fit, default 1000.
        Saves fitting results in .fit_pars, including all posterior distributions.
        Creates .diff_flux and .diff_flux_err (1sigma uncertainty) that includes the differential flux in units of ph/cm2/s/keV for all energy bin
        """

        # init arrays
        self.fit_pars = []
        self.diff_flux = np.zeros(self.dataset.energies.n_energy_bins)
        self.diff_flux_err = np.zeros((self.dataset.energies.n_energy_bins,2))
        self.diff_flux_err2 = np.zeros((self.dataset.energies.n_energy_bins,2))
        
        # loop over energy bins
        for i in tqdm(range(self.dataset.energies.n_energy_bins),desc='Loop over energy bins:'):

            if self.verbose:
                print('###################################################################')
                print('\nStart fitting energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+'...')

            # fit including sky
            if not self.bg_only:

                # sample full posterior
                fit = self.model.sampling(data=self.data_per_energy_bin[i],
                                          chains=1,iter=iters,n_jobs=-1,verbose=False,
                                          pars=pars)

                if self.verbose:
                    print('Summary for energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+':\n')
                    print(fit.stansummary(['flux','Abg']))

                # append fitting results
                self.fit_pars.append(fit)

                # calculate fluxes
                # median value es representative for spectrum
                self.diff_flux[i] = np.percentile(self.fit_pars[i]['flux'],50)/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm

                # 1sigma error bars
                # upper boundary uncertainty
                self.diff_flux_err[i,1] = np.percentile(self.fit_pars[i]['flux'],50+68.3/2)/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm - self.diff_flux[i]
                # lower boundary uncertainty
                self.diff_flux_err[i,0] = np.percentile(self.fit_pars[i]['flux'],50-68.3/2)/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm - self.diff_flux[i]

                # 2sigma error bars
                # upper boundary uncertainty
                self.diff_flux_err2[i,1] = np.percentile(self.fit_pars[i]['flux'],50+95.4/2)/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm - self.diff_flux[i]
                # lower boundary uncertainty
                self.diff_flux_err2[i,0] = np.percentile(self.fit_pars[i]['flux'],50-95.4/2)/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm - self.diff_flux[i]

            # BG-only fit
            else:

                # sample full posterior
                fit = self.model.sampling(data=self.data_per_energy_bin[i],
                                          chains=1,iter=iters,n_jobs=-1,verbose=False,
                                          pars=['Abg','model_tot','model_bg'])

                if self.verbose:
                    print('Summary for energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+':\n')
                    print(fit.stansummary(['Abg']))
                    
                # append fitting results
                self.fit_pars.append(fit)

            if self.verbose:
                print('###################################################################')
            
            
    def TS_map(self,grid):
    """
    TS: still experimental
    Creating a test statistics map from optimising a grid of point source positions.
    :param: grid: 2D grid of longitude/latitude coordinates to test above a background-only fit.

    Output: .TS_bg_only and .TS_vals that include the absolute values of all likelihoods.
    TS: will need to include plotting routine for illustrate the results of this call
    """

        # tested grid
        self.grid = grid

        # make a BG-only run
        bg_only_results = fit(self.dataset,
                              self.pointings,
                              self.response,
                              self.background,
                              bg_only=True)

        # BG-only optimisation
        bg_only_results.MAP_solution()

        # save BG-only result
        self.TS_bg_only = bg_only_results.fit_pars_map[0]['value']

        # init array for saving results
        self.TS_vals = np.zeros(len(self.grid[0].ravel()))

        # loop over grid points
        for i in tqdm(range(len(self.grid[0].ravel())),desc='Loop over grid points:'):

            if self.verbose:
                print(self.grid[0].ravel()[i],self.grid[1].ravel()[i])

            # calculate response for current position
            self.response.calculate_PS_response(self.dataset,
                                                self.pointings,
                                                self.grid[0].ravel()[i],self.grid[1].ravel()[i],self.response.flux_norm,
                                                background=self.background)

            # if something with response entries == zero und dann checken ob es ausserhalb des FoV ist

            # fitting object for current source position
            tmp_results = fit(self.dataset,
                              self.pointings,
                              self.response,
                              self.background)

            try:
                # fit this position
                tmp_results.MAP_solution()
                # save result if not failed
                self.TS_vals[i] = tmp_results.fit_pars_map[0]['value']
                
            except RuntimeError:
                # if fit failed (zeros or something else) through RuntimeError
                if self.verbose:
                    print('Something went wrong with the fit, ignoring grid point ',self.grid[0].ravel()[i],self.grid[1].ravel()[i])
                self.TS_vals[i] = np.nan
        
        

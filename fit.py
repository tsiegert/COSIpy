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
    
    def __init__(self,dataset,pointings,response,background,bg_only=False,priors=None):
        self.dataset = dataset
        self.pointings = pointings
        self.response = response
        self.background = background
        self.count_data = self.reduce_dataset_CDS()
        
        self.bg_only = bg_only

        self.priors = priors

        if not self.bg_only:
            self.data_per_energy_bin = self.make_dictionaries_for_stan()
            self.load_stan_model()
        else:
            print('This will be a background only fit.')
            self.data_per_energy_bin = self.make_dictionaries_for_stan_bg_only()
            self.load_stan_model_bg_only()
    
    
    def reduce_dataset_CDS(self):
        count_data_reduced = []
        for i in range(self.dataset.energies.n_energy_bins):
            
            # reshape count data to reduce CDS if possible
            # this combines the 3 CDS angles into a 1D array for all times at the chosen energy
            yp_tmp = self.dataset.binned_data[:,i,:,:].reshape(self.dataset.times.n_time_bins,
                                                          self.background.bg_model.shape[2]*self.background.bg_model.shape[3])
    
            # reshape count data grid the same way as backgrorund and choose only non-zero indices
            yp_tmp = yp_tmp[:,self.background.calc_this[i]]

            count_data_reduced.append(yp_tmp)
            
        return count_data_reduced
    
    
    def make_dictionaries_for_stan(self):

        all_dicts = []
        
        for i in range(self.dataset.energies.n_energy_bins):
            Np, Nrsp = self.background.bg_model_reduced[i].shape         # initialise sizes of arrays
            N = Np*Nrsp                      # total number of data points

            Nsky = 1

            
            if self.priors == None:
                
                mu_flux_scl = np.array([1.])    # prior centroids for sky, we use 10 because we are ignorant;
                # this has to be an array because it could be more than one
                sigma_flux_scl = np.array([1.]) # same for the width (so, easily 0 but also high values possible)

            else:

                mu_flux_scl = np.array([self.priors[0]])
                sigma_flux_scl = np.array([self.priors[1]])
                
            mu_Abg = 1.                     # for the moment set to a useful value if bg model is ~normalised to data
            sigma_Abg = 1e4                   # same


            # dictionary for data set and prior
            data2D = dict(N = Nrsp,
                          Np = Np,
                          Nsky = Nsky,
                          Ncuts = self.background.Ncuts,
                          bg_cuts = self.background.bg_cuts,
                          bg_idx_arr = self.background.idx_arr,
                          y = self.count_data[i].ravel().astype(int),
                          bg_model = self.background.bg_model_reduced[i],
                          conv_sky = self.response.sky_response[i].reshape(Nsky,Np,Nrsp), # this has to be reshaped because it could be more than one
                          #conv_sky = np.zeros((0,24,1689)),
                          mu_flux = mu_flux_scl,
                          sigma_flux = sigma_flux_scl,
                          mu_Abg = mu_Abg,
                          sigma_Abg = sigma_Abg)
            all_dicts.append(data2D)
            
        return all_dicts
    
    
    def make_dictionaries_for_stan_bg_only(self):
        
        all_dicts = []
        
        for i in range(self.dataset.energies.n_energy_bins):
            Np, Nrsp = self.background.bg_model_reduced[i].shape         # initialise sizes of arrays
            N = Np*Nrsp                      # total number of data points

            mu_Abg = 1.                     # for the moment set to a useful value if bg model is ~normalised to data
            sigma_Abg = 100.                   # same

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
        self.fit_pars_map = []
        self.diff_flux_map = np.zeros(self.dataset.energies.n_energy_bins)
        # loop over energy bins
        for i in tqdm(range(self.dataset.energies.n_energy_bins),desc='Loop over energy bins:'):
            print('Start optimising energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+'...')
            # not necessary in general, but good for quicklook MAP estimate
            init = {}
            if not self.bg_only:
                init['flux'] = np.array([guess])
                init['Abg'] = np.repeat(1.0,self.background.Ncuts)
            else:
                init['Abg'] = np.repeat(1.5,self.background.Ncuts)
            #start = time.time()
            op = self.model.optimizing(data=self.data_per_energy_bin[i],
                                       verbose=False,init=init,as_vector=False,algorithm=method)#,tol_rel_grad=1e5)
            #print(time.time()-start)
            self.fit_pars_map.append(op)
            if not self.bg_only:
                self.diff_flux_map[i] = self.fit_pars_map[i]['par']['flux']/self.dataset.energies.energy_bin_wid[i]/2*self.response.flux_norm
            
            
    def fit(self,iters=1000,pars=['flux','Abg']):
        """
        Fitting COSIpy fit object of a data set with pointing definition, background model, and (now only) point source rersponse.
        Fitting background only is only possible when object is initialised with bg_only = True.
        :option: par    Parameters to save in fit object: pars=['flux','Abg','model_tot','model_bg','model_sky','ppc'], default pars=['flux','Abg'],
                        i.e. no models will be save, only the fitted parameters.
        :option: iters  Number of iterations for fit, default 1000.
        
        """
        self.fit_pars = []
        self.diff_flux = np.zeros(self.dataset.energies.n_energy_bins)
        self.diff_flux_err = np.zeros((self.dataset.energies.n_energy_bins,2))
        self.diff_flux_err2 = np.zeros((self.dataset.energies.n_energy_bins,2))
        # loop over energy bins
        for i in tqdm(range(self.dataset.energies.n_energy_bins),desc='Loop over energy bins:'):
        #for i in tqdm(range(5)):
            print('###################################################################')
            print('\nStart fitting energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+'...')
            #start = time.time()
            if not self.bg_only:
                fit = self.model.sampling(data=self.data_per_energy_bin[i],
                                          chains=1,iter=iters,n_jobs=-1,verbose=False,
                                          pars=pars)
                print('Summary for energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+':\n')
                print(fit.stansummary(['flux','Abg']))
                #print(time.time()-start)
                self.fit_pars.append(fit)
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
                
            else:
                fit = self.model.sampling(data=self.data_per_energy_bin[i],
                                          chains=1,iter=iters,n_jobs=-1,verbose=False,
                                          pars=['Abg','model_tot','model_bg'])
                print('Summary for energy bin '+str(i+1)+'/'+str(self.dataset.energies.n_energy_bins)+':\n')
                print(fit.stansummary(['Abg']))
                #print(time.time()-start)
                self.fit_pars.append(fit)
            print('###################################################################')
            
            
    def TS_map(self,grid):
        
        self.grid = grid
        
        bg_only_results = fit(self.dataset,
                              self.pointings,
                              self.response,
                              self.background,
                              bg_only=True)
        
        bg_only_results.MAP_solution()
        
        self.TS_bg_only = bg_only_results.fit_pars_map[0]['value']
        
        self.TS_vals = np.zeros(len(self.grid[0].ravel()))
        
        for i in tqdm(range(len(self.grid[0].ravel())),desc='Loop over grid points:'):
            
            print(self.grid[0].ravel()[i],self.grid[1].ravel()[i])
            
            self.response.calculate_PS_response(self.dataset,
                                                self.pointings,
                                                self.grid[0].ravel()[i],self.grid[1].ravel()[i],self.response.flux_norm,
                                                background=self.background)

            # if something with response entries == zero und dann checken ob es ausserhalb des FoV ist
            
            tmp_results = fit(self.dataset,
                              self.pointings,
                              self.response,
                              self.background)

            try:
            
                tmp_results.MAP_solution()
                self.TS_vals[i] = tmp_results.fit_pars_map[0]['value']
                
            except RuntimeError:
                
                print('Something went wrong with the fit, ignoring grid point ',self.grid[0].ravel()[i],self.grid[1].ravel()[i])
                self.TS_vals[i] = np.nan
        
        

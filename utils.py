#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----           load libraries           ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pickle
import os

import glob
from dask_jobqueue import PBSCluster
from dask.distributed import Client

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel,ConstantKernel
from sklearn.gaussian_process.kernels import ConstantKernel, RBF
from sklearn.metrics.pairwise import linear_kernel as Linear

from sklearn.metrics import make_scorer
from sklearn.preprocessing import MinMaxScaler 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from scipy.stats import norm
from scipy.fft import fft

import panel as pn
import param
pn.extension()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  server request to aid processing  ----
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def get_cluster(account,cores=30):    
    """Spin up a dask cluster.

    Keyword arguments:
    account -- your account number, e.g. 'UCSB0021'
    cores -- the number of processors requested

    Returns:
    client -- can be useful to inspect client.cluster or run client.close()
    """

    cluster = PBSCluster(
    # The number of cores you want
    cores=1,
    # Amount of memory
    memory='10GB',
     # How many processes
    processes=1,
    # The type of queue to utilize (/glade/u/apps/dav/opt/usr/bin/execcasper)
    queue='casper', 
    # Use your local directory
    local_directory = '$TMPDIR',
    # Specify resources
    resource_spec='select=1:ncpus=1:mem=10GB',
    # Input your project ID here
    account = account,
    # Amount of wall time
    walltime = '02:00:00',
    )

    # Scale up
    cluster.scale(cores)
    
    # Setup your client
    client = Client(cluster)

    return client



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     cluster reading function       ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# modify the function if you want to pass the parameter
# pull 20 years of data
# this has cool potential to have time period subset option
# indexing the selected lists or using multiples of 500 
def read_all_simulations(var):
    '''Prepare cluster list and read to create ensemble(group of data)
    Use preprocess to select only certain dimension and a variable'''
    # Define the list of lists
    cluster_lists = [
        sorted(glob.glob('/glade/campaign/cgd/tss/projects/PPE/PPEn11_LHC/transient/hist/PPEn11_transient_LHC[0][0-5][0-9][0-9].clm2.h0.1995-02-01-00000.nc'))[1:],
        sorted(glob.glob('/glade/campaign/cgd/tss/projects/PPE/PPEn11_LHC/transient/hist/PPEn11_transient_LHC[0][0-5][0-9][0-9].clm2.h0.2000-02-01-00000.nc'))[1:],
        sorted(glob.glob('/glade/campaign/cgd/tss/projects/PPE/PPEn11_LHC/transient/hist/PPEn11_transient_LHC[0][0-5][0-9][0-9].clm2.h0.2005-02-01-00000.nc'))[1:],
        sorted(glob.glob('/glade/campaign/cgd/tss/projects/PPE/PPEn11_LHC/transient/hist/PPEn11_transient_LHC[0][0-5][0-9][0-9].clm2.h0.2010-02-01-00000.nc'))[1:]
    ]
    
    def preprocess(ds, var):
        '''using this function in xr.open_mfdataset as preprocess
        ensures that when only these four things are selected 
        before the data is combined'''
        return ds[['lat', 'lon', 'time', var]]
    
    #read the list and load it for the notebook
    ds = xr.open_mfdataset( cluster_lists, 
                            combine='nested',
                           # lambda allows us to call the predefined preprocess on the ds
                            preprocess = lambda ds: preprocess(ds, var),
                            parallel= True, 
                            concat_dim= ["time", "ens"])
    return ds


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     load data stored in casper     ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----  Gridcell Landareas Data     -----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# reading, storing, subsetting
landarea_file = '/glade/campaign/cgd/tss/projects/PPE/helpers/sparsegrid_landarea.nc'

landarea_ds = xr.open_dataset(landarea_file)

landarea = landarea_ds['landarea']
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----        Parameter Data.       ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# store the parameter names to index later
param_names = {
        key.upper(): value for key, value in {
            'FUN_fracfixers': 0, 'KCN': 1, 'a_fix': 2, 'crit_dayl': 3, 'd_max': 4, 'fff': 5,
            'froot_leaf': 6, 'fstor2tran': 7, 'grperc': 8, 'jmaxb0': 9, 'jmaxb1': 10, 'kcha': 11,
            'kmax': 12, 'krmax': 13, 'leaf_long': 14, 'leafcn': 15, 'lmr_intercept_atkin': 16,
            'lmrha': 17, 'lmrhd': 18, 'medlynintercept': 19, 'medlynslope': 20, 'nstem': 21,
            'psi50': 22, 'q10_mr': 23, 'slatop': 24, 'soilpsi_off': 25, 'stem_leaf': 26,
            'sucsat_sf': 27, 'theta_cj': 28, 'tpu25ratio': 29, 'tpuse_sf': 30, 'wc2wjb0': 31
        }.items()
    }


def param_wrangling():
    # x parameter data for assessment
    df = pd.read_csv('/glade/campaign/asp/djk2120/PPEn11/csvs/lhc220926.txt',index_col=0)
    # the only dimension here is the 'member' aka file index id [LCH0001-500]
    # convert to xr.ds
    params = xr.Dataset(df)

    # list comprehension no need for empty list
    columns = [params[v].values for v in params.data_vars]


    # iterate over params
    return np.array(columns).T


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----     correct time-parsing bug       ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def fix_time(da):
    
    '''fix CESM monthly time-parsing bug'''
    
    yr0 = str(da['time.year'][0].values)
    
    da['time'] = xr.cftime_range(yr0,periods=len(da.time),freq='MS',calendar='noleap')
    
    return da


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  weigh dummy landarea by gridcell  ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----  Weight Gridcells by Landarea ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def weight_landarea_gridcells(da,landarea):

    # weigh landarea variable by mean of gridcell dimension
    return da.weighted(landarea).mean(dim = 'gridcell')

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----       weight var data time dim     ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#------Weighted Averages by Time---
def yearly_weighted_average(da):
    # Get the array of number of days from the main dataset
    days_in_month = da['time.daysinmonth']

    # Multiply each month's data by corresponding days in month
    weighted_month = (days_in_month*da).groupby("time.year").sum(dim = 'time')

    # Total days in the year
    days_per_year = days_in_month.groupby("time.year").sum(dim = 'time')

    # Calculate weighted average for the year
    return weighted_month / days_per_year
    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----    Subset User Selection Funct     ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def subset_var_cluster(var):
    '''Subset the selected variable 
    (s) between 2005-2010 [for now, will be time range]
    as a xr.da.'''
    
    # Read in and wrangle user selected variable cluster
    da_v = read_all_simulations(var).compute()
    # feb. ncar time bug
    da = fix_time(da_v)
    # convert xr.ds to xr.da
    da = da[var]

    return da.compute()


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----      Subset Var Wrangle Funct      ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def wrangle_var_cluster(da):
    '''Weight gridcell dimension by landarea 
    and weight time dimension by the days per each month
    over the total number of days in a year. Globally average
    the selected variable between 2005-2010 
    [for now, will be time range]
    as a xr.da.'''
    # weight gridcell dim by global land area
    da_global = weight_landarea_gridcells(da, landarea)
    # weight time dim by days in month
    da_global_ann = yearly_weighted_average(da_global)
    # take global avg for variable over year dimension
    var = da_global_ann.mean(dim='year')
    
    return var



#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----        Wrangle Data          ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def read_n_wrangle(param, var):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----            Parameter Data.          ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # store user-inputs as global variables
    # will call later for plotting
    global param_name, var_name
    param_name = param
    var_name = var
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----            Parameter Data.          ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # pull in parameter data
    params = param_wrangling()
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    #----        If-else Load Data       ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    
    filepath = os.path.join("preprocessed_data", f"{var}.nc")
    if os.path.exists(filepath):
         #read in the file as a dataset
        ds=xr.open_dataset('preprocessed_data/'+var+'.nc')
    
        #then convert back to data array
        var_avg = ds[var]
    else:
        print(f"Reading and wrangling your data, this may take a few minutes")
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----    Subset User Selection Funct     ----
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        var_da = subset_var_cluster(var)

        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # ----      Subset Var Wrangle Funct      ----
        #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # NEED TO ADD NAME ATTRIBUTE IN WRANGLING PORTION
        var_avg = wrangle_var_cluster(var_da)

        #you ought to convert the data array to dataset before writing to file
        ds = var_avg.to_dataset(name = var)
        ds.to_netcdf('preprocessed_data/'+var+'.nc') # note that this will throw error if you try to overwrite existing files

    return params, var_avg, param_name, var_name





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----        Train Emulator        ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def train_emulator(param, var):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----   Load Pickled Emulation     ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
   # work in progress in GaiaFuture/Scripts/ML/Gaussian/gpr_pickling.ipynb
    # if it's already been queried and saved, pull it!
    # tis only names properly when inside dashboard function
    # commenting out now and adapting bc var is xr.da in this case
    filename = os.path.join("emulation_results", f"gpr_model_{var_name}_{param_name}.sav")
   
    if os.path.exists(filename):
        # load the model from disk
        loaded_model = pickle.load(open(filename, 'rb'))
        
    else:
        print(f"Emulator is running, this may take a few moments")
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----      Split Data 90/10        ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data for splitting
    X_train, X_test, y_train, y_test = train_test_split(param,
                                                        var,
                                                        test_size=0.2,
                                                        # setting a seed
                                                        random_state=0)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----    Kernel Specs No Tuning    ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # initiate the model without tuning
    kernel = ConstantKernel(constant_value = 3,
                            constant_value_bounds=(1e-2, 1e4)) \
                  * RBF(length_scale=1, 
                        length_scale_bounds=(1e-4, 1e8))
   
   
     # using an out of the box kernel for now
    gpr_model = GaussianProcessRegressor(kernel=kernel,
                                        # want 20 random starts
                                        n_restarts_optimizer=20,
                                        # setting seed
                                        random_state=99,
                                        normalize_y=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----         Fit the Model        ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit the model to the training data
    gpr_model = gpr_model.fit(X_train, y_train)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----         Get Predictions      ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Make predictions
    y_pred, y_std = gpr_model.predict(X_test, return_std=True)


    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----         Collect Metrics      ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Verify training score
    #train_score = gpr_model.score(X_train, y_train)

    # Accuracy Score
    #accuracy = accuracy_score(y_test, y_pred)

    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate R^2
    r2_emulator = np.corrcoef(y_test, y_pred)[0,1]**2
    
    # Calculate RMSE
    rmse_emulator = np.sqrt(mean_squared_error(y_test, y_pred))

    # Create a DataFrame to store the results for plotting
    results_df = pd.DataFrame({
        'y_pred': y_pred,
        'y_std': y_std,
        'y_test': y_test,
        'X_test': [x.tolist() for x in X_test],  # Convert array to list for DataFrame
    })

    # Add metrics to the DataFrame
    results_df['R^2'] = r2_emulator
    results_df['RMSE'] = rmse_emulator
    #results_df['Accuracy Score'] = accuracy
    results_df['Mean Absolute Error'] = mae
    
   
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----      Pickle Emulation     ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # save the model to disk
    pickle.dump(gpr_model, open(filename, 'wb')) 
    
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----        Print Metrics         ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Print Training Metrics
    print("R^2:", r2_emulator)
    print("RMSE:", rmse_emulator)
    print("Mean Absolute Error:", mae)
   # print("Training Score:", train_score)
    
    return gpr_model, y_pred, y_std, y_test, X_test, r2_emulator





#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----          Plot Emulator       ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

# Create an array that sets the value of all 32 parameters to 0.5
# this will be used when plotting emulation

X_values = np.full((10, 32), 0.5)  # Fill array with 0.5


def plot_emulator(gpr_model, y_test, r2_emulator):
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----      Visualize Emulation     ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
    # save names
    param_title = param_name.title()
    var_title = var_name.title()

    # index parameter name
    # store the parameter names to index later
    global param_names
    param_names = {
        key.upper(): value for key, value in {
            'FUN_fracfixers': 0, 'KCN': 1, 'a_fix': 2, 'crit_dayl': 3, 'd_max': 4, 'fff': 5,
            'froot_leaf': 6, 'fstor2tran': 7, 'grperc': 8, 'jmaxb0': 9, 'jmaxb1': 10, 'kcha': 11,
            'kmax': 12, 'krmax': 13, 'leaf_long': 14, 'leafcn': 15, 'lmr_intercept_atkin': 16,
            'lmrha': 17, 'lmrhd': 18, 'medlynintercept': 19, 'medlynslope': 20, 'nstem': 21,
            'psi50': 22, 'q10_mr': 23, 'slatop': 24, 'soilpsi_off': 25, 'stem_leaf': 26,
            'sucsat_sf': 27, 'theta_cj': 28, 'tpu25ratio': 29, 'tpuse_sf': 30, 'wc2wjb0': 31
        }.items()
    }
    
    indexed_param = param_names.get(param_name.upper())
    
    # Calculate the z-score for the 99.7% confidence interval
    # 99.7th percentile (three standard deviations)
    z_score = norm.ppf(0.99865)  
    
    
    #For the parameter of interest, replace the 0.5 with a range of values between 0 and 1
    X_values[:, indexed_param] = np.linspace(0, 1, 10)  # Set the 15th column values to evenly spaced values from 0 to 1

    # Predict mean and standard deviation of the Gaussian process at each point in x_values
    y_pred, y_std = gpr_model.predict(X_values, return_std=True)
    # using the emulator R2 value 
    coef_deter = r2_emulator
    #coef_deter = np.corrcoef(y_test[:10],y_pred[:10])[0,1]**2    # DOUBLE CHECK

    
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    plt.plot(X_values[:, indexed_param],
             y_pred[:10,],
             color='#134611',
             label='GPR Prediction')

    plt.text(0.5,np.max(y_pred),
             'R² Score = '+str(np.round(coef_deter,2)),
             fontsize=10)
    
    # applying z-score for 99.7% CI
    plt.fill_between(X_values[:, indexed_param],
                     y_pred[:10] - z_score * y_std[:10], y_pred[:10] + z_score * y_std[:10],
                     alpha=0.5, 
                     color='#9d6b53',
                     label = '3 St.Dev., Confidence Interval')

   #  work in prog, need to adjust according to parameter relationship
   # plt.ylim([0, ])
    plt.xlabel(f'Perturbed Parameter: {param_title}')
    plt.ylabel(f'Variable: {var_title} ')
    plt.title('Parameter Perturbation Uncertainty Estimation')
    
    plt.legend()

     # Save the plot as a PNG file
    plt.savefig(f'plots/emulator/emulator_plot_{var_name}_{param_name}.png')
    
    return plt.show()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----      Plot FAST Accuracy      ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 
def create_parameter_names_dict():
    data = {
        key.upper(): value for key, value in {
            'FUN_fracfixers': 0, 'KCN': 1, 'a_fix': 2, 'crit_dayl': 3, 'd_max': 4, 'fff': 5,
            'froot_leaf': 6, 'fstor2tran': 7, 'grperc': 8, 'jmaxb0': 9, 'jmaxb1': 10, 'kcha': 11,
            'kmax': 12, 'krmax': 13, 'leaf_long': 14, 'leafcn': 15, 'lmr_intercept_atkin': 16,
            'lmrha': 17, 'lmrhd': 18, 'medlynintercept': 19, 'medlynslope': 20, 'nstem': 21,
            'psi50': 22, 'q10_mr': 23, 'slatop': 24, 'soilpsi_off': 25, 'stem_leaf': 26,
            'sucsat_sf': 27, 'theta_cj': 28, 'tpu25ratio': 29, 'tpuse_sf': 30, 'wc2wjb0': 31
        }.items()
    }
    return data

def plot_FAST_accuracy(gpr_model, r2_emulator, y_test, y_pred, y_std):
    def gaussian_regression_lines(gpr_model):
        fourier_amplitudes = []
        
        for param_index in range(32):
            X_values = np.full((10, 32), 0.5)
            X_values[:, param_index] = np.linspace(0, 1, 10)
            y_pred, _ = gpr_model.predict(X_values, return_std=True)
            y_fft = fft(y_pred)
            amplitude = np.abs(y_fft)
            fourier_amplitudes.append(amplitude[1])

        return fourier_amplitudes

    fourier_amplitudes = gaussian_regression_lines(gpr_model)
    sorted_indices = np.argsort(fourier_amplitudes)
    sorted_fourier_amplitudes = np.array(fourier_amplitudes)[sorted_indices]
    swapped_param_keys = {v: k for k, v in create_parameter_names_dict().items()}
    sorted_parameter_names = [swapped_param_keys[index] for index in sorted_indices]

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.barh(range(len(sorted_fourier_amplitudes)), sorted_fourier_amplitudes, color='darkolivegreen')
    ax.set_ylabel('')
    ax.set_xlabel('Fourier Amplitude')
    ax.set_title(f'Fourier Amplitude Sensitivity Test (FAST) for {var_name} vs {param_name}')
    ax.set_yticks(range(len(sorted_fourier_amplitudes)), sorted_parameter_names)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    #ax.text(1, 0.5, f'R2_score = {np.round(r2_train,2)}', fontsize=10, transform=ax.transAxes)
    ax.set_aspect('auto', adjustable='box')

    # Create inset for accuracy plot
    ax_inset = inset_axes(ax, width="40%", height="40%", loc='center right')
    ax_inset.errorbar(y_test, y_pred, yerr=3*y_std, fmt="o", color='#134611')
    ax_inset.plot([0, np.max(y_test)], [0, np.max(y_pred)], linestyle='--', c='k')
    ax_inset.set_xlim([np.min(y_test)-1, np.max(y_test)+1])
    ax_inset.set_ylim([np.min(y_pred)-1, np.max(y_pred)+1])
    ax_inset.set_xlabel('Variable Test')
    ax_inset.set_ylabel(f'Emulated Variable: {var_name}')
    ax_inset.set_title(f'Emulator Accuracy: {var_name}')
    ax_inset.text(0.5, 0.1, f'R² Score = {np.round(r2_emulator, 2)}', fontsize=12, \
                  transform=ax_inset.transAxes, horizontalalignment='center', weight='bold')
    
     # Save the plot as a PNG file
    plt.savefig(f'plots/fast_accuracy/fast_acc_plot_{var_name}_{param_name}.png')

    return plt.show()
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----      Dashboard Wrangle       ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ 

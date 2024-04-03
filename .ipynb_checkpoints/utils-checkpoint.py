#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----           load libraries           ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
#modify the function if you want to pass the parameter
def read_all_simulations(var):
    '''prepare cluster list and read to create ensemble(group of data)
    use preprocess to select only certain dimension and a variable'''
    # read all simulations as a list
    cluster_list= sorted(glob.glob('/glade/campaign/cgd/tss/projects/PPE/PPEn11_LHC/transient/hist/PPEn11_transient_LHC[0][0-5][0-9][0-9].clm2.h0.2005-02-01-00000.nc'))
    cluster_list = cluster_list[1:]

    def preprocess(ds, var):
        '''using this function in xr.open_mfdataset as preprocess
        ensures that when only these four things are selected 
        before the data is combined'''
        return ds[['lat', 'lon', 'time', var]]
    
    #read the list and load it for the notebook
    ds = xr.open_mfdataset( cluster_list, 
                                   combine='nested',
                                   preprocess = lambda ds: preprocess(ds, var),
                                   parallel= True, 
                                   concat_dim="ens")
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
    da_v = read_all_simulations(var)
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


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ----  User Selected ML Plotting Funct   ----
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# function to plot a cluster for to build on for ml
def train_emulator(param, var):
    '''Train the emulator based on the selected parameter and variable'''
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----      Split Data 90/10        ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # data for splitting
    X_train, X_test, y_train, y_test = train_test_split(param,
                                                        var,
                                                        test_size=0.2,
                                                       # setting a seed
                                                        random_state=0)

    gpr_model = GaussianProcessRegressor(normalize_y=True)

    gpr_model.fit(X_train, y_train)

    y_pred, y_std = gpr_model.predict(X_test, return_std=True)

    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # ----         Collect Metrics      ----
    #~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Calculate Mean Absolute Error
    mae = mean_absolute_error(y_test, y_pred)

    # Calculate R^2
    r2_train = r2_score(y_test, y_pred)
    # Calculate RMSE
    rmse_train = np.sqrt(mean_squared_error(y_test, y_pred))

    # Create a DataFrame to store the results for plotting
    results_df = pd.DataFrame({
         'y_pred': y_pred,
        'y_std': y_std,
        'y_test': y_test,
         'X_test': [x.tolist() for x in X_test],  # Convert array to list for DataFrame
    })

    # Add metrics to the DataFrame
    results_df['R^2'] = r2_train
    results_df['RMSE'] = rmse_train
    results_df['Mean Absolute Error'] = mae

    # Print Training Metrics
    print("Training R^2:", r2_train)
    print("Training RMSE:", rmse_train)
    print("Mean Absolute Error:", mae)


#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ---- User Selected Panel Plotting Funct ---- # adding this because working to implement into ml wkflw 
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# function to plot a cluster for the dashboard
def cluster_panel_plot(param_avg, var_avg):
    '''Basic plotting to aid with dashboard set up 
    building on currently.'''
    data = pd.DataFrame({'x': param_avg, 'y': var_avg})
    
    # Create scatter plot using hvplot
    hvplot.scatter(
        x='x', y='y', color='#62c900ff', alpha=0.8,
        xlabel=param, ylabel=var, title='2005-2010 Global Average'
    )
    return pn.panel(scatter_plot)
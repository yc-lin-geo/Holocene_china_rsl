#----------------------Define Functions---------------------------

import numpy as np
import torch
import pyro
import pyro.distributions as dist
from scipy import interpolate
import torch
from pyro.infer import Predictive


def mcmc_predict(input_gpr,mcmc,Xnew,thin_index=1):
    '''
    A function to prediction posterior mean and covariance of GP regression model

    ----------Inputs----------
    input_gpr: a pyro GP regression model
    mcmc: a pyro MCMC object
    Xnew: a torch tensor of new input data

    ----------Outputs----------
    full_bayes_mean_mean: a numpy array of posterior mean of GP regression model
    full_bayes_cov_mean: a numpy array of posterior covariance of GP regression model
    full_bayes_std_mean: a numpy array of posterior standard deviation of GP regression model
    '''
    
    def predictive(X_new,gpr):
        y_loc, y_cov = gpr(X_new,full_cov=True)
        pyro.sample("y", dist.Delta(y_loc))
        pyro.sample("y_cov", dist.Delta(y_cov))
        
    Xnew = torch.tensor(Xnew).double()
    thin_mcmc = mcmc.get_samples()
    for i in thin_mcmc:
        thin_mcmc[i] = thin_mcmc[i][::thin_index]

    posterior_predictive = Predictive(predictive, thin_mcmc)
    full_bayes_mean,full_bayes_cov = posterior_predictive.get_samples(Xnew,input_gpr).values()
    full_bayes_mean_mean = full_bayes_mean.mean(axis=0).detach().numpy()
    full_bayes_cov_mean = full_bayes_cov.mean(axis=0).detach().numpy()
    full_bayes_std_mean = np.diag(full_bayes_cov_mean)**0.5
    likelihood_list = []
    noise = np.ones(len(input_gpr.X))*input_gpr.noise.detach().numpy()

    for i in range(len(full_bayes_mean)):
        f = interpolate.interp1d(Xnew,full_bayes_mean[i])
        likelihood_list.append(cal_likelihood(input_gpr.y.detach().numpy(),
                                              noise**0.5,
                                              f(input_gpr.X)))
        
    return full_bayes_mean_mean,full_bayes_cov_mean,full_bayes_std_mean,likelihood_list


def gen_pred_matrix(age,lat,lon):
    '''
    A function to generate an input matrix for Spatio-temporal GP model

    ----------Inputs----------------
    age: a numpy array, age of the prediction points
    lat: a numpy array, latitude of the prediction points
    lon: a numpy array, longitude of the prediction points

    ----------Outputs----------------
    output_matrix: a torch tensor, input matrix for the spatio-temporal GP model
    '''
    age = np.array(age)
    lat = np.array(lat)
    lon = np.array(lon)

    lon_matrix,lat_matrix,age_matrix = np.meshgrid(lon,lat,age)
    
    output_matrix = torch.tensor(np.hstack([age_matrix.flatten()[:,None],lat_matrix.flatten()[:,None],lon_matrix.flatten()[:,None]])).double()
    return output_matrix

def cal_rate_var(test_X,cov_matrix,mean_rsl,difftimestep=200):
    '''A function to caluclate standard deviation of sea-levle change rate (i.e., first derivative of 
    GP).
    ------------------Inputs----------------------------
    test_X: an array of test input values
    cov_matrix: full covariance matrix from GP regression
    mean_rsl: GP regression produced mean RSL prediction
    difftimestep: time period for averaging 
    
    ------------------Outputs---------------------------
    difftimes: time series for the outputs
    rate: averaged sea-level change rate
    rate_sd: averaged sea-level change rate standard deviation
    '''
    
    Mdiff = np.array(np.equal.outer(test_X, test_X.T),dtype=int) - np.array(np.equal.outer(test_X, test_X.T + difftimestep),dtype=int)
    Mdiff = Mdiff * np.equal.outer(np.ones(len(test_X))*1, np.ones(len(test_X)))
    sub = np.where(np.sum(Mdiff, axis=1) == 0)[0]
    Mdiff = Mdiff[sub, :]
    difftimes = np.abs(Mdiff) @ test_X / np.sum(np.abs(Mdiff), axis=1)
    Mdiff = Mdiff / (Mdiff @ test_X.T)[:,None]
    rate_sd = np.sqrt(np.diag(Mdiff @ cov_matrix @ Mdiff.T))
    rate = Mdiff @ mean_rsl
    
    return difftimes,rate, rate_sd


def decompose_kernels(gpr,pred_matrix,kernels):
    '''
    A function to calculate different kernels contribution to final prediction

    ------------------Inputs----------------------------
    gpr: an optimized pyro GP regression model
    pred_matrix: a torch tensor of prediction matrix containing the data points for prediction
    kernels: a list of pyro kernels for decomposition

    ------------------Outputs---------------------------
    output: a list of tuples, each tuple contains the mean and covariance of the prediction for each kernel 
    '''
    N = len(gpr.X)
    M = pred_matrix.size(0)
    f_loc = gpr.y - gpr.mean_function(gpr.X)
    latent_shape = f_loc.shape[:-1]
    loc_shape = latent_shape + (M,)
    f_loc = f_loc.permute(-1, *range(len(latent_shape)))
    f_loc_2D = f_loc.reshape(N, -1)
    loc_shape = latent_shape + (M,)
    v_2D = f_loc_2D
    Kff = gpr.kernel(gpr.X).contiguous()
    if gpr.noise.dim() <=1:
        Kff.view(-1)[:: N + 1] += gpr.jitter + gpr.noise   # add noise to the diagonal
    elif gpr.noise.dim() ==2:
        Kff = Kff + gpr.noise
        Kff.view(-1)[:: N + 1] += gpr.jitter 
    Lff = torch.linalg.cholesky(Kff)
    
    output = []
    for kernel in kernels:
        Kfs = kernel(gpr.X, pred_matrix)
        pack = torch.cat((f_loc_2D, Kfs), dim=1)
        Lffinv_pack = pack.triangular_solve(Lff, upper=False)[0]
        W = Lffinv_pack[:, f_loc_2D.size(1):f_loc_2D.size(1) + M].t()
        Qss = W.matmul(W.t())
        v_2D = Lffinv_pack[:, :f_loc_2D.size(1)]
        loc = W.matmul(v_2D).t().reshape(loc_shape)
        Kss = kernel(pred_matrix)
        cov = Kss - Qss
        output.append((loc, cov))
        
    return output

def cal_misfit(y,y_sigma,prediction):
    
    return np.mean(np.sqrt(((y-prediction)/y_sigma)**2))

def cal_likelihood(y,y_std,pred):
    '''A function used to calcualte log likelihood function for a given prediction.
    This calculation only considers uncertainty in y axis. 
    
    ------------Inputs------------------
    y: reconstructed rsl
    y_std: standard deviation of reconstructed rsl
    pred: mean predction of rsl
    
    ------------Outputs------------------
    likelihood: mean likelihood of prediction fit to observation
    '''
    from scipy.stats import norm

    log_likelihood = 1 
    for i in range(len(y)):
        
        norm_dis = norm(y[i], y_std[i])
        log_likelihood+=np.log(norm_dis.pdf(pred[i]))
    
    return log_likelihood

def cal_MSE(y,yhat):
    '''
    A function to calculate MSE coefficient
    '''
    MSE = np.sum((yhat-y)**2)/len(y)
    return MSE


def cal_wMSE(y,yhat,y_sigma):
    '''
    A function to calculate weighted MSE coefficient
    '''
    wMSE = np.sum((yhat-y)**2/y_sigma**2)/len(y)
    return wMSE
    

def get_change_point_posterior(guide,sample_number):
    num_cp = int(list(guide().keys())[-1][list(guide().keys())[-1].index('_')+1:])
    output_dict = dict()
    output_dict['b'] = np.zeros(sample_number)
    output_dict['a'] = np.zeros([sample_number,num_cp+1])
    output_dict['cp'] = np.zeros([sample_number,num_cp])
    test_cp = []
    for i in range(num_cp):
        test_cp.append(guide.median()['cp_'+str(i)].detach().numpy())
    cp_index = np.argsort(test_cp)
    
    for i in range(sample_number):
        store_beta = []
        store_cp = []
        posterior_samples = guide()
        for i2 in range(num_cp+1):
            store_beta.append(posterior_samples['a_'+str(i2)].detach().numpy())
            if i2 < num_cp:
                store_cp.append(posterior_samples['cp_'+str(i2)].detach().numpy())
        output_dict['b'][i] = posterior_samples['b'].detach().numpy()
        output_dict['a'][i] = np.array(store_beta)
        output_dict['cp'][i] = np.array(store_cp)[cp_index]
    return output_dict

def change_point_forward(n_cp,cp_loc_list,new_X,data_X,beta_coef_list,b):
    '''
    A function to calculate the forward model of the change-point model

    ------------Inputs--------------
    n_cp: int, number of change-points
    cp_loc_list: 1D torch tensor with shape (n_cp), the location of the change-points
    new_X: 2D torch tensor with shape (n_samples,n_features) for new data prediction
    data_X: 2D torch tensor with shape (n_samples,n_features) for training data
    beta_coef_list: 1D torch tensor with shape (n_cp+1), the slope coefficients
    b: float, the intercept coefficient
    '''
    last_intercept = b
    mean = torch.zeros(new_X.shape[0])
    for i in range(n_cp+1):
        if i==0:
            start_age = data_X[:,0].min()
            start_idx = 0
            end_age = cp_loc_list[i]
            end_idx = torch.where(new_X<end_age)[0][-1]+1
            last_change_point = start_age
        elif i==n_cp:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(new_X>=start_age)[0][0]
            end_age = new_X[:,0].max()
            end_idx = new_X.shape[0]
        else:
            start_age = cp_loc_list[i-1]
            start_idx = torch.where(new_X>=start_age)[0][0]
            end_age = cp_loc_list[i]
            end_idx = torch.where(new_X<end_age)[0][-1]+1
        
        mean[start_idx:end_idx] = beta_coef_list[i] * (new_X[start_idx:end_idx:,0]-last_change_point) + last_intercept
        last_intercept = beta_coef_list[i] * (end_age-last_change_point) + last_intercept
        last_change_point = end_age
    return mean



def cal_geo_dist2(X,Z=None):
    '''
    A function to calculate the squared distance matrix between each pair of X.
    The function takes a PyTorch tensor of X and returns a matrix
    where matrix[i, j] represents the spatial distance between the i-th and j-th X.
    
    -------Inputs-------
    X: PyTorch tensor of shape (n, 2), representing n pairs of (lat, lon) X
    R: approximate radius of earth in km
    
    -------Outputs-------
    distance_matrix: PyTorch tensor of shape (n, n), representing the distance matrix
    '''
    if Z is None:
        Z = X

    # Convert coordinates to radians
    X = torch.tensor(X)
    Z = torch.tensor(Z)
    X_coordinates_rad = torch.deg2rad(X)
    Z_coordinates_rad = torch.deg2rad(Z)
    
    # Extract latitude and longitude tensors
    X_latitudes_rad = X_coordinates_rad[:, 0]
    X_longitudes_rad = X_coordinates_rad[:, 1]

    Z_latitudes_rad = Z_coordinates_rad[:, 0]
    Z_longitudes_rad = Z_coordinates_rad[:, 1]

        # Calculate differences in latitude and longitude
    dlat = X_latitudes_rad[:, None] - Z_latitudes_rad[None, :]
    dlon = X_longitudes_rad[:, None] - Z_longitudes_rad[None, :]
    # Apply Haversine formula
    a = torch.sin(dlat / 2) ** 2 + torch.cos(X_latitudes_rad[:, None]) * torch.cos(Z_latitudes_rad[None, :]) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))

    return c**2


def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'

    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]

    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)

def rsl_data_pred(total_rsl_site,obs_age,site_index,pred_age=np.arange(12000,-100,-1000)):
    '''
    A function to predict rsl for each data point given the GIA prediction for each sites

    Inputs:

    ----------------------------
    total_rsl_site: 3D array, The GIA prediction for each sites, the shape should be [number of ice models, number of earth models, number of time steps]
    obs_age: 1D array, The age of the observation data
    site_index: 1D array, The index of the GIA prediction corresponding to each data point
    pred_age: 1D array, The age of the GIA prediction

    Outputs:

    ----------------------------
    tota_rsl_data: 3D array, The GIA prediction for each data point, the shape should be [number of ice models, number of earth models, number of data points]
    '''
    
    tota_rsl_data = np.zeros([*total_rsl_site.shape[:2],len(obs_age)])
    for i2 in range(total_rsl_site.shape[0]):
        for i3 in range(total_rsl_site.shape[1]):
            for i in range(len(obs_age)):
                pred_x,pred_y = pred_age, total_rsl_site[i2,i3,site_index[i],:]
                pred_interp = interpolate.interp1d(pred_x,pred_y)
                tota_rsl_data[i2,i3,i] = pred_interp(age[i])
    return tota_rsl_data

def cal_rsl_misfit(total_rsl_site,obs_x,obs_y,obs_x_uncert,obs_y_uncert,site_index,pred_age=np.arange(12000,-100,-1000)):
    '''
    A function to calculate rsl data-model misfit based on misfit function from Love et al., 2016
    
    Inputs:

    ----------------------------------
    total_rsl_site: 3D array, The GIA prediction for each sites, the shape should be [number of ice models, number of earth models, number of time steps]
    obs_x: 1D array, The age of the observation data
    obs_y: 1D array, The RSL of the observation data
    obs_x_uncert: 1D array, The 1 sigma age uncertainty of the observation data
    obs_y_uncert: 1D array, The 1 sigma RSL uncertainty of the observation data
    site_index: 1D array, The index of the GIA prediction corresponding to each data point
    pred_age: 1D array, The age of the GIA prediction

    Outputs:

    ----------------------------------
    toto_misfit: 2D array, The misfit for each ice and earth model, the shape should be [number of ice models, number of earth models]
    toto_misfit_all: 3D array, The misfit for each ice and earth model for each data point, the shape should be [number of ice models, number of earth models, number of data points]
    '''
    toto_misfit = np.zeros([*total_rsl_site.shape[:2]])
    toto_misfit_all =  np.zeros([*total_rsl_site.shape[:2],len(obs_x)])

    for i2 in range(total_rsl_site.shape[0]):
        for i3 in range(total_rsl_site.shape[1]):
            output_mis = np.zeros(len(obs_x))
            for i in range(len(obs_x)):
                obs_age,obs_rsl = obs_x[i],obs_y[i]
                obs_xerr,obs_yerr = obs_x_uncert[i],obs_y_uncert[i]
                pred_x,pred_y = pred_age, total_rsl_site[i2,i3,site_index[i],:]
                pred_interp = interpolate.interp1d(pred_x,pred_y)
                min_interp_age = obs_age-3*obs_xerr
                if min_interp_age<0: min_interp_age = 0 #GIA prediction only go to 0 BP
                interp_age = torch.arange(min_interp_age,obs_age+3*obs_xerr,1)
                
                pred_y = torch.tensor(pred_interp(interp_age))

                mis_age = (interp_age-obs_age)**2
                mis_rsl = (pred_y-obs_rsl)**2
                min_index = np.argmin(mis_age+mis_rsl)
                output_mis[i] = np.sqrt(mis_rsl[min_index]+mis_age[min_index])
            toto_misfit[i2,i3] = np.mean(output_mis)
            toto_misfit_all[i2,i3] = output_mis
    return toto_misfit,toto_misfit_all
def read_rsl_out(input_file,num_time_step):
    rsl_all,site_coord = [],[]
    rsl_out = open(input_file,'r')
    rsl_out = rsl_out.readlines()
    rsl_time = np.array(rsl_out[0].split(),dtype = float)
    rsl_out = rsl_out[1:]
    for i in range(len(rsl_out)):
        if i%(num_time_step+1) ==0:
            site_coord.append(np.array([float(i) for i in rsl_out[i].split()[1:3]]))
        else:
            rsl_all+=[float(i) for i in rsl_out[i].split()]
    return np.array(rsl_all).reshape([int(len(rsl_all)/num_time_step),num_time_step]),np.array(site_coord),rsl_time

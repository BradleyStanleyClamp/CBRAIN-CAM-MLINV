# tb - 7/8/2022 - Utilities for equation discovery in Python

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_validate
from sklearn.model_selection import RepeatedKFold
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from scipy.integrate import cumtrapz,trapz

def p_derivative(var,pressure):
    """
    Calculates derivative of var with respect to pressure via centered finite differences
    """
    L = pressure.shape[1]
    dvar_dp_left = (var[:,[L-2]]-var[:,[L-1]])/(pressure[:,[L-2]]-pressure[:,[L-1]])
    dvar_dp_mid = (var[:,:L-2]-var[:,2:L])/(pressure[:,:L-2]-pressure[:,2:L])
    dvar_dp_right = (var[:,[1]]-var[:,[0]])/(pressure[:,[1]]-pressure[:,[0]])
    output = np.concatenate((dvar_dp_right,dvar_dp_mid,dvar_dp_left),axis=1)
    
    return output
    
def p_integral(var,pressure,partial,L=30):
    """
    Calculate the integral of var with respect to pressure 
    above (lower values of pressure) and 
    below (higher values of pressure)
    """
    G = 9.8
    L = pressure.shape[1]
    
    if partial == 'above': INT = cumtrapz(x=pressure,y=var,initial=0)/g
    elif partial == 'below': 
        INT = np.outer(trapz(x=pressure,y=var)/g,np.arange(0,30)**0)-\
        cumtrapz(x=pressure,y=var,initial=0)/g
        
    return INT

def p_der_spline(var,pressure_data,pressure_eval,order,kind='cubic'):
    """
    Calculate derivative of var with respect to pressure
    using spline interpolation
    pressure_data = Pressure vector used to take the derivative
    pressure_eval = Pressure vector used to evaluate derivatives
    order = Order of derivative
    kind = Type of spline interpolation
    """
    assert var.shape[0] == pressure_eval.shape[0]
    assert var.shape[0] == pressure_data.shape[0]
    
    output = np.zeros(pressure_eval.shape)
    
    for isample in range(var.shape[0]):
        
        f = interpolate.interp1d(pressure_data[isample,:],var[isample,:],kind,axis=0,fill_value="extrapolate")
        
        if order == 1: der = misc.derivative(f, pressure_eval[isample,:], dx=1e-2, n=1, args=(), order=3)
        elif order == 2: der = misc.derivative(f, pressure_eval[isample,:], dx=1e-2, n=2, args=(), order=5)
        output[isample,:] = der
    
    return output

def subsampler(ind,x,xRH,xB,xLHFns,xt,yt,xRHt,xBt,xLHFnst,hyam,hybm,
               variables=['p','q','dq_dp','dq_dp_FD','d2q_dp2',
                         'd2q_dp2_FD','Q_above','Q_below','T',
                         'dT_dp','d2T_dp2','dT_dp_FD','d2T_dp2_FD',
                         'T_above','T_below','RH','dRH_dp',
                         'dRH_dp_FD','d2RH_dp2','d2RH_dp2_FD',
                         'RH_above','RH_below','B','dB_dp',
                          'd2B_dp2','dB_dp_FD','d2B_dp2_FD',
                          'B_above','B_below','ps','S0','SHF',
                          'LHF','LHFns','dq_dt','dT_dt']):
    """
    Subsamples and formats data from various sources into a 
    dictionary of inputs and outputs used for equation discovery
    ind = Subsampling indices
    x = Input vector containing raw variables
    xRH = Input vector containing relative humidity
    xB = Input vector containing plume buoyancy
    xLHFns = Input vector containing rescaled latent heat fluxes
    xt, yt, xRHt, xBt, xLHFnst = Same but for the test set
    hyam,hybm = Used to calculate the hybrid sigma vertical coordinate
    variables = Variables to include in the formatted dataset
    """
    
    # Sub-sample the training set
    x_sub = x[ind,:]; y_sub = y[ind,:]; 
    xRH_sub = xRH[ind,:]; xB_sub = xB[ind,:]; xLHFns_sub = xLHFns[ind];
    # Sub-sample the test sets
    xt_sub = {}; yt_sub = {}; xRHt_sub = {}; xBt_sub = {}; xLHFnst_sub = {};
    climates = xt.keys()
    for iclim,clim in enumerate(climates):
        xt_sub[clim] = xt[clim][ind,:]
        yt_sub[clim] = yt[clim][ind,:]
        xRHt_sub[clim] = xRHt[clim][ind,:]
        xBt_sub[clim] = xBt[clim][ind,:]
        xLHFnst_sub[clim] = xLHFnst[clim][ind]
    # Pressure
    PS = x_sub[:,60]; pm = P0*hyam+np.outer(PS,hybm)
    # Training structure
    x_train = {}
    if 'p' in variables: x_train['p'] = pm
    if 'q' in varaibles: x_train['q'] = x_sub[:,:30].values
    if 'dq_dp' in variables: x_train['dq_dp'] = p_der_spline(x_train['q'],x_train['p'],x_train['p'],1)
    if 'dq_dp_FD' in variables: x_train['dq_dp_FD'] = p_derivative(x_train['q'],x_train['p'],1)
    if 'd2q_dp2' in variables: x_train['d2q_dp2'] = p_der_spline(x_train['q'],x_train['p'],x_train['p'],2)
    if 'd2q_dp2_FD' in variables: x_train['d2q_dp2_FD'] = p_derivative(x_train['dq_dp'],x_train['p'],1)
    if 'Q_above' in variables: x_train['Q_above'] = p_integral(x_train['q'],x_train['p'],'above')
    if 'Q_below' in variables: x_train['Q_below'] = p_integral(x_train['q'],x_train['p'],'below')
    if 'T' in variables: x_train['T'] = x_sub[:,30:60].values
    if 'dT_dp' in variables: x_train['dT_dp'] = p_der_spline(x_train['T'],x_train['p'],x_train['p'],1)
    if 'd2T_dp2' in variables: x_train['d2T_dp2'] = p_der_spline(x_train['T'],x_train['p'],x_train['p'],2)
    if 'dT_dp_FD' in variables: x_train['dT_dp_FD'] = p_derivative(x_train['T'],x_train['p'],1)
    if 'd2T_dp2_FD' in variables: x_train['d2T_dp2_FD'] = p_derivative(x_train['dT_dp_FD'],x_train['p'],1)
    if 'T_above' in variables: x_train['T_above'] = p_integral(x_train['T'],x_train['p'],'above')
    if 'T_below' in variables: x_train['T_below'] = p_integral(x_train['T'],x_train['p'],'below')
    if 'RH' in variables: x_train['RH'] = xRH_sub.values
    if 'dRH_dp' in variables: x_train['dRH_dp'] = p_der_spline(x_train['RH'],x_train['p'],x_train['p'],1)
    if 'dRH_dp_FD' in variables: x_train['dRH_dp_FD'] = p_derivative(x_train['RH'],x_train['p'],1)
    if 'd2RH_dp2' in variables: x_train['d2RH_dp2'] = p_der_spline(x_train['RH'],x_train['p'],x_train['p'],2)
    if 'd2RH_dp2_FD' in variables: x_train['d2RH_dp2_FD'] = p_derivative(x_train['dRH_dp'],x_train['p'],1)
    if 'RH_above' in variables: x_train['RH_above'] = p_integral(x_train['RH'],x_train['p'],'above')
    if 'RH_below' in variables: x_train['RH_below'] = p_integral(x_train['RH'],x_train['p'],'below')
    if 'B' in variables: x_train['B'] = xB_sub.values
    if 'dB_dp' in variables: x_train['dB_dp'] = p_der_spline(x_train['B'],x_train['p'],x_train['p'],1)
    if 'd2B_dp2' in variables: x_train['d2B_dp2'] = p_der_spline(x_train['B'],x_train['p'],x_train['p'],2)
    if 'dB_dp_FD' in variables: x_train['dB_dp_FD'] = p_derivative(x_train['B'],x_train['p'],1)
    if 'd2B_dp2_FD' in variables: x_train['d2B_dp2_FD'] = p_derivative(x_train['dB_dp_FD'],x_train['p'],1)
    if 'B_above' in variables: x_train['B_above'] = p_integral(x_train['B'],x_train['p'],'above')
    if 'B_below' in variables: x_train['B_below'] = p_integral(x_train['B'],x_train['p'],'below')
    if 'ps' in variables: x_train['ps'] = x_sub[:,60].values
    if 'S0' in variables: x_train['S0'] = x_sub[:,61].values
    if 'SHF' in variables: x_train['SHF'] = x_sub[:,62].values
    if 'LHF' in variables: x_train['LHF'] = x_sub[:,63].values
    if 'LHFns' in variables: x_train['LHFns'] = xLHFns_sub.values
    y_train = {}
    if 'dq_dt' in variables: y_train['dq_dt'] = y_sub[:,:30].values
    if 'dT_dt' in variables: y_train['dT_dt'] = y_sub[:,30:60].values
    # Test structure
    x_test = {}; y_test = {}
    for iclim,clim in enumerate(climates):
        print(clim)
        x_test[clim] = {}; y_test[clim] = {};
        if 'p' in variables: x_test[clim]['p'] = P0*hyam+np.outer(xt_sub[clim][:,60].values,hybm)
        if 'q' in variables: x_test[clim]['q'] = xt_sub[clim][:,:30].values
        if 'dq_dp' in variables: x_test[clim]['dq_dp'] = p_der_spline(x_test[clim]['q'],x_test[clim]['p'],x_test[clim]['p'],1)
        if 'dq_dp_FD' in variables: x_test[clim]['dq_dp_FD'] = p_derivative(x_test[clim]['q'],x_test[clim]['p'],1)
        if 'd2q_dp2' in variables: x_test[clim]['d2q_dp2'] = p_der_spline(x_test[clim]['q'],x_test[clim]['p'],x_test[clim]['p'],2)
        if 'd2q_dp2_FD' in variables: x_test[clim]['d2q_dp2_FD'] = p_derivative(x_test[clim]['dq_dp'],x_test[clim]['p'],1)
        if 'Q_above' in variables: x_test[clim]['Q_above'] = p_integral(x_test[clim]['q'],x_test[clim]['p'],'above')
        if 'Q_below' in variables: x_test[clim]['Q_below'] = p_integral(x_test[clim]['q'],x_test[clim]['p'],'below')
        if 'T' in variables: x_test[clim]['T'] = xt_sub[clim][:,30:60].values
        if 'dT_dp' in variables: x_test[clim]['dT_dp'] = p_der_spline(x_test[clim]['T'],x_test[clim]['p'],x_test[clim]['p'],1)
        if 'dT_dp_FD' in variables: x_test[clim]['dT_dp_FD'] = p_derivative(x_test[clim]['T'],x_test[clim]['p'],1)
        if 'd2T_dp2' in variables: x_test[clim]['d2T_dp2'] = p_der_spline(x_test[clim]['T'],x_test[clim]['p'],x_test[clim]['p'],2)
        if 'd2T_dp2_FD' in variables: x_test[clim]['d2T_dp2_FD'] = p_derivative(x_test[clim]['dT_dp'],x_test[clim]['p'],1)
        if 'T_above' in variables: x_test[clim]['T_above'] = p_integral(x_test[clim]['T'],x_test[clim]['p'],'above')
        if 'T_below' in variables: x_test[clim]['T_below'] = p_integral(x_test[clim]['T'],x_test[clim]['p'],'below')
        if 'RH' in variables: x_test[clim]['RH'] = xRHt_sub[clim].values
        if 'dRH_dp' in variables: x_test[clim]['dRH_dp'] = p_der_spline(x_test[clim]['RH'],x_test[clim]['p'],x_test[clim]['p'],1)
        if 'dRH_dp_FD' in variables: x_test[clim]['dRH_dp_FD'] = p_derivative(x_test[clim]['RH'],x_test[clim]['p'],1)
        if 'd2RH_dp2' in variables: x_test[clim]['d2RH_dp2'] = p_der_spline(x_test[clim]['RH'],x_test[clim]['p'],x_test[clim]['p'],2)
        if 'd2RH_dp2_FD' in variables: x_test[clim]['d2RH_dp2_FD'] = p_derivative(x_test[clim]['dRH_dp'],x_test[clim]['p'],1)
        if 'RH_above' in variables: x_test[clim]['RH_above'] = p_integral(x_test[clim]['RH'],x_test[clim]['p'],'above')
        if 'RH_below' in variables: x_test[clim]['RH_below'] = p_integral(x_test[clim]['RH'],x_test[clim]['p'],'below')
        if 'B' in variables: x_test[clim]['B'] = xBt_sub[clim].values
        if 'dB_dp' in variables: x_test[clim]['dB_dp'] = p_der_spline(x_test[clim]['B'],x_test[clim]['p'],x_test[clim]['p'],1)
        if 'dB_dp_FD' in variables: x_test[clim]['dB_dp_FD'] = p_derivative(x_test[clim]['B'],x_test[clim]['p'],1)
        if 'd2B_dp2' in variables: x_test[clim]['d2B_dp2'] = p_der_spline(x_test[clim]['B'],x_test[clim]['p'],x_test[clim]['p'],2)
        if 'd2B_dp2_FD' in variables: x_test[clim]['d2B_dp2_FD'] = p_derivative(x_test[clim]['dB_dp'],x_test[clim]['p'],1)
        if 'B_above' in variables: x_test[clim]['B_above'] = p_integral(x_test[clim]['B'],x_test[clim]['p'],'above')
        if 'B_below' in variables: x_test[clim]['B_below'] = p_integral(x_test[clim]['B'],x_test[clim]['p'],'below')
        if 'ps' in variables: x_test[clim]['ps'] = xt_sub[clim][:,60].values
        if 'S0' in variables: x_test[clim]['S0'] = xt_sub[clim][:,61].values
        if 'SHF' in variables: x_test[clim]['SHF'] = xt_sub[clim][:,62].values
        if 'LHF' in variables: x_test[clim]['LHF'] = xt_sub[clim][:,63].values
        if 'LHFns' in variables: x_test[clim]['LHFns'] = xLHFnst_sub[clim].values
        if 'dq_dt' in variables: y_test[clim]['dq_dt'] = yt_sub[clim][:,:30].values
        if 'dT_dt' in variables: y_test[clim]['dT_dt'] = yt_sub[clim][:,30:60].values
        
    return x_train,x_test,y_train,y_test

def range_normalizer(x_train,x_test,scalar_keys,vector_keys,Norm=None):
    """
    (Optionally) calculates a normalization structure (Norm)
    and normalizes input variables using their range
    x_train = Training set, used to calculate normalization structure
    x_test = Test set
    scalar_keys = Keys corresponding to scalar variables
    vector_keys = Keys corresponding to vector variables
    """
    if Norm==None:
        Norm = {}
        Norm['mean'] = {}
        Norm['min'] = {}
        Norm['max'] = {}
        Norm['std'] = {}
        for keys in combin_keys:
            Norm['mean'][keys] = np.mean(x_train[keys].flatten())
            Norm['min'][keys] = np.min(x_train[keys].flatten())
            Norm['max'][keys] = np.max(x_train[keys].flatten())
            Norm['std'][keys] = np.std(x_train[keys].flatten())
    
    combin_keys = np.concatenate((scalar_keys,vector_keys))
    x_train_range = {}
    x_test_range = {}
    for key in scalar_keys:
        x_train[key] = np.outer(x_train[key],np.ones((30,)))
    for iclim,clim in enumerate(climates):
        for key in scalar_keys:
            x_test[clim][key] = np.outer(x_test[clim][key],np.ones((30,)))
    for key in combin_keys: x_train_range[key] = \
        (x_train[key].flatten()-Norm['min'][key])/(Norm['max'][key]-Norm['min'][key])
    for iclim,clim in enumerate(climates):
        x_test_range[clim] = {}
        for key in combin_keys:
            x_test_range[clim][key] = \
            (x_test[clim][key].flatten()-\
             Norm['min'][key])/(Norm['max'][key]-Norm['min'][key])
            
    return x_train_range,x_test_range,Norm

def dic_to_array(key_array,x_train_range,x_test_range,scale_dict):
    """
    Converts dictionary (x_train_range,x_test_range) containing 
    normalized variables into array that can be used for regression
    key_array = Keys to use in the regression problem
    scale_dict = Normalization for the ouput variables (e.g., to physical units)
    """
    # Initialization
    X_train = np.zeros((x_train_range['p'].shape[0],len(key_array)))
    X_test = {};
    for iclim,clim in enumerate(climates):
        X_test[clim] = np.zeros((x_test_range[clim]['p'].shape[0],len(key_array)))
    
    # Input Assignments
    for ikey,key in enumerate(key_array):
        X_train[:,ikey] = x_train_range[key]
        for iclim,clim in enumerate(climates):
            X_test[clim][:,ikey] = x_test_range[clim][key]
    
    # Output Assignments
    dTdt_train = (y_train['dT_dt'] * scale_dict['TPHYSTND']).flatten()
    dQdt_train = (y_train['dq_dt'] * scale_dict['PHQ']).flatten()
    dTdt_test = {}; dQdt_test = {};
    for iclim,clim in enumerate(climates):
        dTdt_test[clim] = (y_test[clim]['dT_dt'] * scale_dict['TPHYSTND']).flatten()
        dQdt_test[clim] = (y_test[clim]['dq_dt'] * scale_dict['PHQ']).flatten()
        
    return X_train,X_test,dQdt_train,dQdt_test,dTdt_train,dTdt_test

def SFS_poly(features,X_train,X_test,Y_train,Y_test,min_features,max_features):
    """
    Sequential feature selection based on polynomial fits
    features = Inputs to select
    X_train = Training input array
    X_test = Test input array
    Y_train = Training output array
    Y_test = Test output array
    min_features = Minimal number of features to use for feature selection
    max_features = Maximal number of features to use for feature selection
    """
    dict_output = {}
    lin_reg = LinearRegression()
    for no_features in np.arange(min_features, max_features):   
        sfs = SequentialFeatureSelector(lin_reg,
                                        n_features_to_select=no_features,
                                        direction='forward',
                                        cv=cv[ideg], n_jobs=-1)
        sfs.fit(X_train, Y_train)
        selected_features = np.array(features)[sfs.get_support()].tolist()
        X_transformed = sfs.transform(X_train)
        lin_reg.fit(X_transformed, Y_train)
        ypred = lin_reg.predict(X_transformed)
        mse_train = mean_squared_error(Y_train, ypred)
        mse_test = {}; ypred_test = {};
        for iclim,clim in enumerate(climates):
            ypred_test[clim] = lin_reg.predict(sfs.transform(X_test[clim]))
            mse_test[clim] = mean_squared_error(Y_test[clim], ypred_test[clim])

        # Output dictionary
        dict_exp = {}
        for i in range(len(selected_features)):
            dict_exp[selected_features[i]] = lin_reg.coef_[i]
        dict_exp['LR_Bias'] = lin_reg.intercept_
        dict_exp['mse_train'] = mse_train
        dict_exp['mse_test'] = mse_test
        print(dict_exp,'\n')
        dict_output['Number of variables %d'%no_features] = dict_exp
    
    return dict_output
#!/usr/bin/env python

import numpy as np
from mh import *

def regress_out(Y, conf):    
    mean = Y.mean()
    Y    = Y-mean
    conf = np.array(conf)
    conf = conf-np.mean(conf,axis=0)
    if conf.ndim == 1:
        conf = conf.reshape(-1, 1)
    beta = np.linalg.pinv(conf).dot(Y)
    Y = Y - conf * beta + mean #reg.intercept_    
    return Y

# Prepare data
def prepare(df,name=None,Y=None,deconfound=None,normalise=True,exclude=None):
    if exclude is not None:
        df.drop(df.index[exclude])
    # Data and regressors
    if name is not None:
        Y = np.array(df[name])
        nan_check = np.isnan(Y)
        if np.count_nonzero(nan_check) > 0:
            print("Warning: NaNs found in the data; cleaning those up...")
            Y = Y[~nan_check]

    b = np.array(df['age_birth'][~nan_check])
    s = np.array(df['age_scan'][~nan_check])
    if deconfound is not None:
        # Confounds
        nan_check = np.isnan(np.array(df[deconfound]))
        conf = np.array(df[[deconfound]])  # For struct data
        
        Y = regress_out(Y[~nan_check],conf[~nan_check])
        b = np.array(df['age_birth'][~nan_check])
        s = np.array(df['age_scan'][~nan_check])
    if normalise:
        # Normalise to 95th percentile
        Y = Y/np.quantile(Y,.95,axis=0)*100  

    return Y,b,s
             
   
# Various forward models
prem_thresh = 37 # prematurity threshold
# Only one slope (beta1=beta2)
def forward_0(p,t_birth,t_scan):
    beta0,t_onset = p
    return -beta0*t_onset+beta0*t_scan

# Same post-birth slopes for term and prem
def forward_1(p,t_birth,t_scan):
    beta0, beta1, t_onset = p
    return -beta0*t_onset+beta0*t_birth+beta1*(t_scan-t_birth)

# Post-birth slope is different
def forward_2(p,t_birth,t_scan):
    beta0, beta1_term, beta1_prem, t_onset = p
    
    term = t_birth>=prem_thresh
    prem = t_birth<prem_thresh
    
    pred = np.zeros(t_scan.size)
    pred[term] = -beta0*t_onset+beta0*t_birth[term]+beta1_term*(t_scan[term]-t_birth[term])
    pred[prem] = -beta0*t_onset+beta0*t_birth[prem]+beta1_prem*(t_scan[prem]-t_birth[prem])
    
    return pred

# All slopes different - same onset
def forward_3(p,t_birth,t_scan):
    beta0_term, beta1_term, beta0_prem, beta1_prem, t_onset = p
    
    term = t_birth>=prem_thresh
    prem = t_birth<prem_thresh
    
    pred = np.zeros(t_scan.size)
    pred[term] = -beta0_term*t_onset+beta0_term*t_birth[term]+beta1_term*(t_scan[term]-t_birth[term])
    pred[prem] = -beta0_prem*t_onset+beta0_prem*t_birth[prem]+beta1_prem*(t_scan[prem]-t_birth[prem])
    
    return pred


models_list = [ 
    {
        'forward' : forward_0, 
        'params'  : ['beta0','t_onset'] , 
        'bounds'  : ([-np.inf,0],[np.inf,40.0]),
        'init'    : [0,0.00001]
    },
    {
        'forward' :forward_1, 
        'params'  : ['beta0','beta1','t_onset'], 
        'bounds'  : ([-np.inf,-np.inf,0],[np.inf,np.inf,40.0]), 
        'init'    : [0,0,0.00001]
        
    },
    
    {
        'forward' :forward_2, 
        'params'  : ['beta0','beta1-term','beta1-prem','onset'], 
        'bounds'  : ([-np.inf,-np.inf,-np.inf,0],[np.inf,np.inf,np.inf,40.0]),
        'init'    : [0,0,0,0.00001]
    },
    
    {
        'forward' :forward_3, 
        'params'  : ['beta0-term','beta1-term','beta0-prem','beta1-prem','onset'], 
        'bounds'  : ([-np.inf,-np.inf,-np.inf,-np.inf,0],[ np.inf, np.inf, np.inf, np.inf,40]),
        'init'    : [0,0,0,0,0.00001]
    }
]
               

class ForwardModel:
    def __init__(self,modelid):
        self.modelid = modelid
        self.forward = models_list[self.modelid]['forward']
        self.labels  = models_list[self.modelid]['params']       
        self.nparams = len(self.labels)
        
    def bounds(self):
        LB = np.array(models_list[self.modelid]['bounds'])[0]
        UB = np.array(models_list[self.modelid]['bounds'])[1]
        return LB, UB
    
    def init(self):   
        return np.array( models_list[self.modelid]['init'] )
    
    def get_params(self,params,group):
        onset = params[-1] #np.mean(samples[:,-1],axis=0)
        if self.modelid == 1:
            beta0 = params[0] #np.mean(samples[:,0],axis=0)
            beta1 = params[1] #np.mean(samples[:,1],axis=0)
        if self.modelid == 2:        
            beta0 = params[0] #np.mean(samples[:,0],axis=0)
            if group == 'term':
                beta1 = params[1] #np.mean(samples[:,1],axis=0)
            else:
                beta1 = params[2] #np.mean(samples[:,2],axis=0)
        if self.modelid == 3:
            if group == 'term':
                beta0 = params[0] #np.mean(samples[:,0],axis=0)
                beta1 = params[1] #np.mean(samples[:,1],axis=0)
            else:
                beta0 = params[2] #np.mean(samples[:,2],axis=0)
                beta1 = params[3] #np.mean(samples[:,3],axis=0)
   
        return beta0,beta1,onset

        
        
# Fit model to data
def do_fit(Y,t_birth,t_scan,forward_model):
    loglik  = lambda p : np.log(np.linalg.norm(Y-forward_model.forward(p,t_birth,t_scan)))*Y.size/2
    logpr   = lambda p : 0 #*np.sum(gauss_logpr(p[:-1],loc=0,scale=40))
    # Bounds
    LB,UB = forward_model.bounds()
    # Initialise
    p0   = forward_model.init()

    mh = MH(loglik,logpr,njumps=10000)
    #import time
    #start = time.time()
    samples = mh.fit(p0,LB=LB,UB=UB)
    ML      = mh.marglik_Laplace(samples)
    
    return samples, ML

# SVD the data prior to fitting (for vertex-wise data)
def do_pca_fit(Y,b,s,forward_model,keep=10):
    
    if Y.shape[0]>Y.shape[1]:
        raise Exception("Data must be transposed")
    import scipy as sp
    #U,S,V = sp.sparse.linalg.svds(Y-Y.mean(axis=0),k=keep)
    U,S,V = sp.sparse.linalg.svds(Y,k=keep)
    
    all_betas = np.zeros((fm.nparams,keep))
    for i in range(keep):
        samples, _ = do_fit(Y@V[i,:].T,b,s,forward_model)
        betas = samples[:,:-1].mean(axis=0)
        betas = np.append(betas,-samples[:,-1].mean(axis=0)*betas[0])
        all_betas[:,i] = betas
    all_betas = all_betas@V
    grot1 = all_betas[:-1,:]
    grot2 = -all_betas[-1,:]/all_betas[0,:]
    all_betas = np.concatenate((grot1,grot2[None,:]))
    return all_betas



def extract_long(df):
    tmp = df.copy() #loc[df['age_birth']>30]
    long_1 = tmp.loc[ tmp['scanned_twice'] >0]
    long_1 = long_1.loc[long_1['scanned_twice'] <100]
    long_2 = tmp.loc[ tmp['scanned_twice'] >100]
    return long_1,long_2


# Plotting
def plot_model_prediction(b1,b2,o,birth=37,scan=45,col='b'):
    import matplotlib.pyplot as plt
    @np.vectorize
    def forward(t,birth,beta1,beta2,onset):    
        c0    = -onset*beta1
        c1    = (beta1-beta2)*birth-onset*beta1
        y = 0
        if t<onset:
            y=0
        elif onset<=t<birth:
            y = c0+beta1*t
        else:
            y = c1+beta2*t
        return y

    weeks_pre = np.arange(0,birth)
    weeks_post = np.arange(birth,scan)
    y_weeks_pre  = forward(weeks_pre,birth,b1,b2,o)
    y_weeks_post  = forward(weeks_post,birth,b1,b2,o)
    y_onset  = forward(o,birth,b1,b2,o)
    y_birth  = forward(birth,birth,b1,b2,o)    
    h = plt.plot(weeks_pre,y_weeks_pre,'k-')
    plt.plot(weeks_post,y_weeks_post,'-',color=col)
    plt.plot(o,y_onset,'ko')
    plt.plot(birth,y_birth,'ko')

    return h

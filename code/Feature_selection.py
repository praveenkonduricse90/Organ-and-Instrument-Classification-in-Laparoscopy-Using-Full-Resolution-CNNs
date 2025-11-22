import numpy as np
from numpy.random import *
from Fitness_Calculation import *
from math import *

def Transfer_functionv4(x):
    tfv4 = (2*tanh((pi*x)/2))/pi
    return tfv4

def V_tranfer_fun(data):
    binared_data = np.zeros((data.shape))
    for i in range(data.shape[0]):
        tf = data[i]
        value = np.array([Transfer_functionv4(vl) for vl in tf])  
        av = np.average(value)
        value[value > av] = 1   
        value[value <= av] = 0
        binared_data[i]=value
    return binared_data

def AQUILA(data_,lab):
    # % Parameters
    lb    = 0  ; ub   = 1;
    LB    = 0  ; UB   = 1;
    alpha = 0.1;delta = 0.1;
    w     = 0.9
    max_it=10;
    thres = 0.5;
    # % Number of dimensions
    dim = data_.shape[1];
    # % Initial 
    uni = np.unique(lab)
    ta_ind = list()
    for u in uni:
        wh = np.where(lab==u)[0]
        ta_ind.extend(wh[:100])
    data = data_[ta_ind,:]
    label = lab[ta_ind]
    N  = data.shape[0] ;
    X = V_tranfer_fun(data)
    # % Fitness
    fitG =float('inf');        
    ws = [0.99, 0.01];
    fitb =list()  
    for k in range(N):
        a1=X[k,:].copy()
        a1[a1>thres]=1
        a1[a1<thres]=0
        b=np.where(a1 == 1)
        selfea=data[:,b[0]]
        if len(selfea[0]) != 0:
            fitc = Fitness_function(selfea,label,data,ws);
            fitb.append(fitc)
            # Gbest update
            if fitc < fitG:
                Xgb  = X[k,:]; 
                fitG = fitc;
    n_ft =[randint(int(dim/4),int(dim/1.5)) for x in range(1)]
    # PBest
    Xnew  = X.copy(); 
    t=1  
    rr = 0.3
    while t <= max_it:
        G2    = 2*rr-1; # 
        G1    = 2*(1-(t/max_it)); # 
        to    = dim;
        u     = .0265;
        r0    = 10;
        r     = r0 +u*to;
        omega = .005;
        phi0  = 3*np.pi/2;
        phi   = -omega*to+phi0;
        x     = r* np.sin(phi);  #
        y     = r* np.cos(phi);  # 
        QF    = pow(t,((2*rr-1)/(1-max_it)**2));
        
        for i in range(N):
            if t<=(2/3)*max_it:
                if np.amin(Xnew[i,:])<0.5:
                    Xnew[i,:]=Xgb*(1-t/max_it)+(np.mean(X[i,:])-Xgb)*rr; #
                else:
                    Xnew[i,:]=Xgb+ Xnew[i,:] + (y-x)*rr; #
            else:
                if np.amin(Xnew[i,:])<0.5:
                    Xnew[i,:]=(Xgb-np.mean(X[i,:]))*alpha-rr+((UB-LB)*rr+LB)*delta; #%
                else:
                    Xnew[i,:]=QF*Xgb-(G2*X[i,:]*rr)-G1+rr*G2; # 
            # % Fitness
            a1=Xnew[i,:].copy()
            a1[a1>thres]=1
            a1[a1<thres]=0
            b=np.where(a1 == 1)
            selfea=data[:,b[0]]
            if len(selfea[0]) != 0:
                fit = Fitness_function(selfea,label,data,ws);
                fitb.append(fit)
                # Gbest update
                if fit < fitG:
                  Xgb  = Xnew[i,:];
                  fitG = fit;
            Xnew[Xnew > ub]=ub
            Xnew[Xnew < lb]=lb
        
        t=t+1;
    a11=Xgb
    a11[a11>thres]=1
    a11[a11<thres]=0
    SF=np.where(a11 == 1)[0]
    if len(SF)<r0:S_I = sol_chk(a11,n_ft[0])
    else :S_I = SF
    return S_I




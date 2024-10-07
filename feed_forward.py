import numpy as np 
import matplotlib.pyplot as plt
from utils import abline_2 
from utils import relu 

def feedfoward_NN(X,W,b,set_groups,plot=False):

    color='green'
    ### Proyetion
    W_proy=W['W1']
    b1=b['b1']
    if plot:
        abline_2(W_proy[:,0], b1,-1,r'$H_1^1$')  
        plt.scatter(X[:,0],X[:,1], c=color)
        varianx = (np.max(X[:,0])-np.min(X[:,0]))/4
        variany = (np.max(X[:,1])-np.min(X[:,1]))/4        
        plt.xlim([np.min(X[:,0]-varianx),np.max(X[:,0]+varianx)])
        plt.ylim([np.max(X[:,1]-variany)-4,np.max(X[:,1]+variany)+1])
        plt.show()            
    X1=relu(np.dot(X,W_proy)+b1)      
    if plot:
        plt.scatter(X1, np.zeros_like(X1), c=color)
        plt.show()          
    W2=W['W2']
    b2=b['b2']
    X2=relu(np.dot(X1,W2)+b2)
    if plot:
        abline_2(np.array([W2[0,0],0]), b2[0],1,r'$H_2^1$')    
        abline_2(np.array([W2[0,1],0]), b2[1],-1,r'$H_2^2$')  
        plt.scatter(X1, np.zeros_like(X1), c=color)
        plt.xlim([np.min(X1[:,0]-0.01),np.max(X1[:,0]+0.01)])
        plt.ylim([-0.1,0.3])
        plt.show()
    ### Iteration 

    ########### actualizamos grupos
    Xi=X2  
    for i in range(len(set_groups)-3):
        W_loc = W['W'+str(i+3)]
        b_loc = b['b'+str(i+3)]
        Xi_n = relu(np.dot(Xi,W_loc)+b_loc)
        if plot:
            abline_2(W_loc[:,0], b_loc[0],1,r'$H_{'+str(i+3)+'}^1$')    
            abline_2(W_loc[:,1], b_loc[1],-1,r'$H_{'+str(i+3)+'}^2$')  
            plt.scatter(Xi[:,0],Xi[:,1], c=color)
            varianx = (np.max(Xi[:,0])-np.min(Xi[:,0]))/4
            variany = (np.max(Xi[:,1])-np.min(Xi[:,1]))/4        
            plt.xlim([-varianx,np.max(Xi[:,0]+varianx)])
            plt.ylim([-variany,np.max(Xi[:,1]+variany)])
            plt.show()
        Xi = Xi_n
##### FINAL CLASIFICICATION ##### 
    if plot:
        plt.scatter(Xi[:,0],Xi[:,1])
    N = len(set_groups)-1
    W_loc = W['W'+str(N+1)]
    b_loc = b['b'+str(N+1)]
    Xi_n = relu(np.dot(Xi,W_loc)+b_loc)
    if plot:
        plt.scatter(Xi[:,0],Xi[:,1], c=color)
        abline_2(np.array([W_loc[0],W_loc[1]]), b_loc,-1,r'$H_{'+str(N+1)+'}^1$')    
        varianx = (np.max(Xi[:,0])-np.min(Xi[:,0]))/4
        variany = (np.max(Xi[:,1])-np.min(Xi[:,1]))/4        
        plt.xlim([-varianx,np.max(Xi[:,0]+varianx)])
        plt.ylim([-variany,np.max(Xi[:,1]+variany)])
        plt.show()
    Xi = Xi_n
    if plot:
        plt.scatter(Xi, np.zeros_like(Xi), c=color)
        plt.show()  
    XN = Xi
    Y_est = np.array([])
    for i in XN:
        Y_est=np.append(Y_est, i*(-1)+1)
    return Y_est
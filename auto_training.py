import numpy as np 
import matplotlib.pyplot as plt
import os        # To obtain the path of the Python document

from utils import abline_2 
from utils import relu 
from utils import check_folder 



def auto_train(X,Y,plot=True,dimension=2, save_images=False, example='example_1'):
    colors = np.array(['#B3001E', '#1E599C'])
    W={}
    b={}  
    Dic={}

    fig = plt.figure(dpi=150,figsize=(8,6))
    plt.scatter(X[:,0],X[:,1],c=colors[Y])
    varianx = (np.max(X[:,0])-np.min(X[:,0]))/4
    variany = (np.max(X[:,1])-np.min(X[:,1]))/4        
    plt.xlim([np.min(X[:,0]-varianx),np.max(X[:,0]+varianx)])
    plt.ylim([np.max(X[:,1]-variany)-4,np.max(X[:,1]+variany)+1])
    plt.title('Input data')
    if save_images:
        newpath = check_folder(example)
        name = os.path.join(newpath,'images_0.png')
        plt.savefig(name, dpi=150 )
    if plot:
        plt.show()  
    else:
        plt.close()  

    
    ### Projection
    np.random.seed(5)
    W_proy=np.random.rand(dimension)*5
    W_proy=W_proy.reshape(dimension,1)
    b1=5
    for i in range(X.shape[0]):
        if np.dot(X[i],W_proy)+b1<0:
            print('b is no big enought.')
        for j in range(X.shape[0]):
            if np.dot(X[j],W_proy)==np.dot(X[i],W_proy) and i!=j:
                print('Same proyection with '+ '['+str(round(X[i,0],2)) +','+str(round(X[i,1],2))+
          '], and ['+str(round(X[j,0],2)) +','+str(round(X[j,1],2))+']. Here i='+str(i)+', and j='+str(j))
                break

    fig = plt.figure(dpi=150,figsize=(8,6))
    axis = plt.gca()
    abline_2(W_proy.reshape(-1,2)[0], b1,1,axis,r'$H_1$')  
    plt.scatter(X[:,0],X[:,1],c=colors[Y])
    varianx = (np.max(X[:,0])-np.min(X[:,0]))/4
    variany = (np.max(X[:,1])-np.min(X[:,1]))/4        
    plt.xlim([np.min(X[:,0]-varianx),np.max(X[:,0]+varianx)])
    plt.ylim([np.max(X[:,1]-variany)-4,np.max(X[:,1]+variany)+1])
    plt.title('Layer 1')
    if save_images:
        newpath = check_folder(example)
        name = os.path.join(newpath,'images_1.png')
        plt.savefig(name, dpi=150 )
    if plot:
        plt.show()  
    else:
        plt.close()  


    Dic['X0']=X
    W['W1']=W_proy
    b['b1']=b1

    X1=relu(np.dot(X,W_proy)+b1)    
           

    #### Order the data

    loc_list=[]
    for i in range(X1.shape[0]):
        loc_list.append(np.array([X1[i][0],Y[i],i]))
    to_sort=np.array(loc_list)
    sort=to_sort[np.lexsort(to_sort.T[::-1])]
    X1=sort[:,0].reshape(X1.shape[0],1)
    Y=sort[:,1].astype(int)
    order=sort[:,2]
    
    Dic['X1']=X1

    ### Define groupes 

    set_groups={'G0': np.array([X1[0]])}
    j=0
    for i in range(X1.shape[0]-1):
        if Y[i]==Y[i+1]:
            set_groups['G'+str(j)]=np.append( set_groups['G'+str(j)] , X1[i+1])
        else:
            j+=1
            set_groups['G'+str(j)]=np.array([X1[i+1]])

    #### Initialization 

    b21 = -(set_groups['G1'].max()+set_groups['G2'].min())/2
    b22 = (set_groups['G0'].max()+set_groups['G1'].min())/2
    

    w21=[1]
    w22=[-1]

    W2=np.array([w21,w22]).T
    b2=np.array([b21,b22]).T
    
    X2=relu(np.dot(X1,W2)+b2)

    W['W2']=W2
    b['b2']=b2
    Dic['X2']=X2
    fig = plt.figure(dpi=150,figsize=(8,6))
    axis = plt.gca()
    abline_2(np.array([w21[0],0]), b21,1,axis,r'$H_2^1$')    
    abline_2(np.array([w22[0],0]), b22,-1,axis,r'$H_2^2$')  
    plt.scatter(X1, np.zeros_like(X1),c=colors[Y])
    plt.xlim([np.min(X1[:,0]-0.01),np.max(X1[:,0]+0.01)])
    plt.ylim([-0.1,0.3])
    plt.title('Layer 2')

    if save_images:
        newpath = check_folder(example)
        name = os.path.join(newpath,'images_2.png')
        plt.savefig(name, dpi=150 )
    if plot:
        plt.show()  
    else:
        plt.close()

    ### Iteration 

    ########### actualizamos grupos

    Xi=X2
    for i in range(len(set_groups)-3):

        dif=0
        set_groups_n={}
        ## Redefine coordenandas de elementos en los grupos
        for j in range(len(set_groups)):
            set_groups_n['G'+str(j)]=Xi[dif:dif+len(set_groups['G'+str(j)])]
            dif+=len(set_groups['G'+str(j)])
        set_groups=set_groups_n
        
        a0 =  (set_groups['G'+str(i)][:,1].min()/2 + set_groups['G'+str(i)][:,1].max())
        a1 =  (set_groups['G'+str(i)][:,1].min()/2)

        a2 =  (set_groups['G'+str(i+2)][:,0].min()/2)
        a3 =  ((set_groups['G'+str(i+2)][:,0].max() + set_groups['G'+str(i+3)][:,0].min())/2)




        w1_loc = np.array([a0/a3, 1])
        w2_loc = np.array([-a1/a2, -1])
        b1_loc = -a0
        b2_loc = a1

        W_loc=np.array([w1_loc,w2_loc]).T
        b_loc=np.array([b1_loc,b2_loc]).T

        W['W'+str(i+3)]=W_loc
        b['b'+str(i+3)]=b_loc
        


        Xi_n=relu(np.dot(Xi,W_loc)+b_loc)
        
        Dic['X'+str(i+1)]=Xi

        
        fig = plt.figure(dpi=150,figsize=(8,6))
        axis = plt.gca()
        abline_2(w1_loc, b1_loc,1,axis,r'$H_{'+str(i+3)+'}^1$')    
        abline_2(w2_loc, b2_loc,-1,axis,r'$H_{'+str(i+3)+'}^2$')  
        plt.scatter(Xi[:,0],Xi[:,1],c=colors[Y])
        varianx = (np.max(Xi[:,0])-np.min(Xi[:,0]))/4
        variany = (np.max(Xi[:,1])-np.min(Xi[:,1]))/4        
        plt.xlim([-varianx,np.max(Xi[:,0]+varianx)])
        plt.ylim([-variany,np.max(Xi[:,1]+variany)])
        plt.title('Layer '+str(i+3))

        if save_images:
            newpath = check_folder(example)
            name = os.path.join(newpath,'images_'+str(i+3)+'.png')
            plt.savefig(name, dpi=150 )
        if plot:
            plt.show()  
        else:
            plt.close()

        Xi=Xi_n


    ##### FINAL CLASIFICICATION ##### 

    set_groups_n={}
    dif=0
    ## Redefine coordenandas de elementos en los grupos
    for j in range(len(set_groups)):
        set_groups_n['G'+str(j)]=Xi[dif:dif+len(set_groups['G'+str(j)])]
        dif+=len(set_groups['G'+str(j)])
    set_groups=set_groups_n

    N=len(set_groups)-1

    a0 =  (set_groups['G'+str(N-2)][:,1].min()/2)
    a1 =  (set_groups['G'+str(N)][:,0].min()/2)

    w1_loc = -1/a1
    w2_loc = -1/a0
    b1_loc = 1

    W_loc=np.array([w1_loc,w2_loc]).T
    b_loc=b1_loc



    Xi_n=relu(np.dot(Xi,W_loc)+b_loc)

    W['W'+str(N+1)]=W_loc
    b['b'+str(N+1)]=b_loc
    Dic['X'+str(N+1)]=Xi_n
    
    fig = plt.figure(dpi=150,figsize=(8,6))
    axis = plt.gca()
    abline_2(np.array([w1_loc,w2_loc]), b1_loc,-1,axis,r'$H_{'+str(N+1)+'}^1$')    
    plt.scatter(Xi[:,0],Xi[:,1],c=colors[Y])
    varianx = (np.max(Xi[:,0])-np.min(Xi[:,0]))/4
    variany = (np.max(Xi[:,1])-np.min(Xi[:,1]))/4        
    plt.xlim([-varianx,np.max(Xi[:,0]+varianx)])
    plt.ylim([-variany,np.max(Xi[:,1]+variany)])
    plt.title('Layer '+str(N+1))
    if save_images:
        newpath = check_folder(example)
        name = os.path.join(newpath,'images_'+str(N+1)+'.png')
        plt.savefig(name, dpi=150 )
    if plot:
        plt.show()  
    else:
        plt.close()
        
    Xi=Xi_n
    
    fig = plt.figure(dpi=150,figsize=(8,6))
    axis = plt.gca()
    plt.scatter(Xi, np.zeros_like(Xi),c=colors[Y])
    plt.title('Output data')

    if save_images:
        newpath = check_folder(example)
        name = os.path.join(newpath,'images_'+str(N+2)+'.png')
        plt.savefig(name, dpi=150 )
    if plot:
        plt.show()  
    else:
        plt.close()


    set_groups_n={}
    dif=0
    ## Redefine coordenandas de elementos en los grupos
    for j in range(len(set_groups)):
        set_groups_n['G'+str(j)]=Xi[dif:dif+len(set_groups['G'+str(j)])]
        dif+=len(set_groups['G'+str(j)])
    set_groups=set_groups_n
    
    loc_list=[]
    for i in range(Xi.shape[0]):
        loc_list.append(np.array([order[i],Xi[i]]))
    to_sort=np.array(loc_list)
    sort=to_sort[np.lexsort(to_sort.T[::-1])]
    XN=sort[:,1].reshape(Xi.shape[0],1)
    
    Y_est=np.array([])
    for i in XN:

        Y_est=np.append(Y_est, i[0])
    
    return Y_est, W, b, set_groups, Dic
import numpy as np 
import matplotlib.pyplot as plt
import os        # To obtain the path of the Python document
import imageio     # To make the gif

def relu(x):
    return x * (x > 0)


def abline_2(w, b,a,axes,labels=None):
    colors=["#F7B538","#0F4C5C"]
    if w[1]==0:
        w_loc=0.0001
    else:
        w_loc=w[1]
    slope = -w[0]/w_loc
    intercept= -b/w_loc

    axes.set_xlim(-200,2000)
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    X_square=np.array([[(-b-3000*w[1])/w[0],3000],[(-b+3000*w[1])/w[0],-3000],[a*3000,0]])
    plt.fill(X_square[:,0], X_square[:,1],alpha=0.4, c=colors[int(labels[-2])-1])
    plt.plot(x_vals, y_vals, '--',linewidth=2,label=labels, c=colors[int(labels[-2])-1])
    plt.legend(fontsize=10)
    del x_vals
    del y_vals

def gen_data(n,d=-0.35):
    np.random.seed(0)
    a11=np.ones(n)*np.random.rand(n)
    a12=np.ones(n)*np.random.rand(n)
    a21=np.ones(n)*np.random.rand(n)
    a22=np.ones(n)*np.random.rand(n)

    a1=np.concatenate((a11,a21)).reshape(n,2)
    a2=np.concatenate((a12,a22)).reshape(n,2)+[0,d]

    c1=np.ones(n,dtype=int)
    c2=np.zeros(n,dtype=int)

    X=np.concatenate((a1,a2))
    Y=np.concatenate((c1,c2))
    return X,Y


def check_folder(example):
    cwd = os.getcwd()
    newpath = os.path.join(os.path.join(cwd,"images"),"images_"+example) 
    if not os.path.exists(newpath): #We create the a folder called gif
        os.makedirs(newpath)
    return newpath


def figure_names(n,example):
    names_list = []
    newpath = check_folder(example)
    print(newpath)
    for i in range(n+1):
        names_list.append(os.path.join(newpath,'images_'+str(i+1)+'.png'))
    return names_list


def make_gif(example,duration=5):

    png_dir = check_folder(example)
    images = []
    for file_name in sorted(os.listdir(png_dir)):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
            
    # Make it pause at the end so that the viewers can ponder
    for _ in range(10):
        images.append(imageio.imread(file_path))
    
    cwd = os.getcwd()
    newpath_gif = os.path.join(cwd,"gifs") 
    if not os.path.exists(newpath_gif): #We create the a folder called gif
        os.makedirs(newpath_gif)
    
    imageio.mimsave(newpath_gif+'/'+example+'.gif', images, format='GIF', duration=duration)
    print('Saved gif')
    return None
    

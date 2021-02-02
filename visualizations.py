import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path
import matplotlib.animation as animation



def video2d(samples, title, time, fps=60, xrange=[-6,6], yrange=[-6,6]):
    
    #Make subplots
    fig, ((ax1, ax2),(ax3,ax4)) = plt.subplots(2,2,figsize=(15,15))
    fig.title=title
    
    #Total number of samples used
    numSamples = samples.shape[0]*samples.shape[2]
    
    #x, y samples
    xSamples = np.reshape(samples[:,0,:], samples.shape[0]*samples.shape[2])
    ySamples = np.reshape(samples[:,1,:], samples.shape[0]*samples.shape[2])
    ax3.axis("off")
    
    ims=[]
    frames = int(fps*time) #Number of frames in the video
    maxVal=1
    xMax=1
    yMax=1
    for i in range(frames):
        #Save 2d hitos
        data ,_,_ = np.histogram2d(xSamples[:(i*numSamples)//frames],
                                   ySamples[:(i*numSamples)//frames],
                                   range=[xrange, yrange], density=True,
                                   bins=50)
        #Figure out the ranges for the final graphs
        if(i==frames-1):
            maxVal = data.max()
            n, _ = np.histogram(xSamples[:(i*numSamples)//frames], 1000,
                                density=True, range=xrange)
            xMax = n.max()
            n, _ = np.histogram(ySamples[:(i*numSamples)//frames], 1000,
                                density=True, range=xrange)            
            yMax = n.max()
            
        ims.append(data)
    
    #Initial histo for the x projection
    data = xSamples[:]
    n, bins = np.histogram(data, 1000, density=True, range=xrange)
    
    # get the corners of the rectangles for the histogram
    top = np.array(bins[:-1])
    bottom = np.array(bins[1:])
    right = np.zeros(len(top))
    left = right + n
    nrects = len(top)
    
    #Draw lines between the rectangle corners
    nverts = nrects * (1 + 3 + 1)
    verts = np.zeros((nverts, 2))
    codes = np.ones(nverts, int) * path.Path.LINETO
    codes[0::5] = path.Path.MOVETO
    codes[4::5] = path.Path.CLOSEPOLY
    verts[0::5, 0] = right
    verts[0::5, 1] = top
    verts[1::5, 0] = left
    verts[1::5, 1] = top
    verts[2::5, 0] = left
    verts[2::5, 1] = bottom
    verts[3::5, 0] = right
    verts[3::5, 1] = bottom
    
    patch = None
    
    
    
    #Initial histo for the y projection
    data2 = ySamples[:]
    n2, bins2 = np.histogram(data2, 1000, density=True, range=yrange)
    
    # get the corners of the rectangles for the histogram
    left2 = np.array(bins2[:-1])
    right2 = np.array(bins2[1:])
    bottom2 = np.zeros(len(left2))
    top2 = bottom2 + n2
    nrects2 = len(left2)
    
    #Draw lines between the rectangle corners
    nverts2 = nrects2 * (1 + 3 + 1)
    verts2 = np.zeros((nverts2, 2))
    codes2 = np.ones(nverts2, int) * path.Path.LINETO
    codes2[0::5] = path.Path.MOVETO
    codes2[4::5] = path.Path.CLOSEPOLY
    verts2[0::5, 0] = left2
    verts2[0::5, 1] = bottom2
    verts2[1::5, 0] = left2
    verts2[1::5, 1] = top2
    verts2[2::5, 0] = right2
    verts2[2::5, 1] = top2
    verts2[3::5, 0] = right2
    verts2[3::5, 1] = bottom2
    
    patch2 = None
    
    
    #First 2d histo
    a = ims[0]
    im = ax2.imshow(a, interpolation='none', aspect='auto', vmin=0,
                    vmax=maxVal*1.15, origin="lower",
                    extent=[-6,6,-6,6])
    
    #First x histo
    barpath = path.Path(verts, codes)
    patch = patches.PathPatch(
        barpath, facecolor='blue', edgecolor='blue', alpha=1)
    ax1.add_patch(patch)
    
    ax1.set_ylim(xrange)
    ax1.set_xlim(xMax*1.3,0)
    
    
    #First y histo
    barpath2 = path.Path(verts2, codes2)
    patch2 = patches.PathPatch(
        barpath2, facecolor='blue', edgecolor='blue', alpha=1)
    ax4.add_patch(patch2)
    
    ax4.set_xlim(yrange)
    ax4.set_ylim(yMax*1.3, 0)
    
    
    graphs=[im, patch, patch2]
    
    
    #Draw the new graphs upon each iteration
    def animate(i):
        if i % fps == 0: #Indicate movement
            print( '.', end ='' )
        
        # simulate new data coming in
        data = xSamples[:(i*numSamples)//frames]
        n, bins = np.histogram(data, 1000, density=True, range=xrange)
        left = right + n
        verts[1::5, 0] = left
        verts[2::5, 0] = left
        
        data2 = ySamples[:(i*numSamples)//frames]
        n2, bins2 = np.histogram(data2, 1000, density=True, range=yrange)
        top2 = bottom2 + n2
        verts2[1::5, 1] = top2
        verts2[2::5, 1] = top2
        
        graphs[0].set_array(ims[i])
    
        return graphs
        
    #Animator object
    anim = animation.FuncAnimation(fig, 
                                   animate,
                                   frames, repeat=False, blit=True)
    
    #Save the animation as a .mp4
    anim.save(title+'.mp4', fps=fps, extra_args=['-vcodec', 'libx264'])
    
    print('Done!')
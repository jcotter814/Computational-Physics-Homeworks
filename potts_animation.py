# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 17:06:17 2019
@author: James Cotter
Potts model movie.
"""

import potts_model as P
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['animation.ffmpeg_path'] = r'C:\Users\James Cotter\Desktop\School work\Senior Year, Semester 1\Computational Physics\ffmpeg-20191208-9f7b2b3-win64-static\bin\ffmpeg.exe'
import matplotlib.animation as animation


nT = 40 # number of steps in temperature

Temps = np.flip(np.linspace(0.1, 1.5, nT), 0) # temperatures to evaluate
#Temps = np.linspace(0.5, 3.0, 10) * nT

stepsPerFrame = 100 # how many MC steps to take per movie frame
framesPerTemp = 40 # how many frames to record at each temperature

# initialize the model (4 states)
model = P.Potts(100,100,3)

# create the initial movie object
fig, ax = plt.subplots()    
im = ax.imshow(model.S, cmap=plt.get_cmap('jet'))
ax = plt.gca()
ax.set_aspect(1)
plt.xlabel('x')
plt.ylabel('y')
title = plt.title('T=%4.3f' % 0.0)
plt.tight_layout()


# animation function that drives the Ising simulation
def anim(i):
    
    # find out the current temperature from the frame index
    tempIndex = i // framesPerTemp
    T = Temps[tempIndex]
    
    # find initial equilibrium
    if i==0:
        for j in range(1000):    
            model.mcStep(T)
    
    # run MC steps
    for j in range(stepsPerFrame):
        model.mcStep(T)
    
    # update
    im.set_array(model.S)
    title.set_text("T = %4.3f" % T)
    
    print("T = %4.3f, frame = %d" % (T, i % framesPerTemp))
    
    #plt.pause(0.01)
    
    return im, title

# compose the animation
frameCount = len(Temps) * framesPerTemp


ani = animation.FuncAnimation(fig, anim, interval=10, frames=range(frameCount), blit=False)

writer = animation.FFMpegWriter(fps=30, bitrate=4000)
ani.save("potts3.mp4", writer=writer)


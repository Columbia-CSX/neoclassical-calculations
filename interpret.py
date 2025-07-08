import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
from scipy.optimize import root
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser(description="process input tags")
parser.add_argument('--rootChoice', type=str, required=True, help='low middle or high, which root to use (if only two, use low or middle for electron root)')

rootChoice = parser.parse_args().rootChoice


print("~~~ Interpretting results from runRadErScan ~~~")
files = os.listdir('./')

rNs = []
datas = []
for file in files:
    if file.endswith('Jr-vs-Er.dat'):
        datas.append(np.transpose(np.loadtxt(file)))
        rNs.append(float(file.split('_')[1].split('-')[0]))

rNs, datas = zip(*sorted(zip(rNs, datas), key=lambda pair: pair[0]))
rNs = list(rNs)
datas = list(datas)
print("Data for the following rN values has been succesfully retrieved:")
print(rNs)

def plot_data(data, ErRange=None):
    Er_data = data[0]
    Jr_data = data[1]

    Jr = interp1d(Er_data, Jr_data, kind='linear')
    
    if ErRange is None:
        ErRange = np.max(abs(Er_data))

    Er = np.linspace(-ErRange, ErRange, 1000)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    if np.max(Jr_data)/1e-3 > 100:
        ax.set_yscale('symlog', linthresh=1e-3)
    plt.plot(Er_data, Jr_data, linestyle=None, marker='o')
    plt.plot(Er, Jr(Er))
    fig.show()

def deriv(y, x0):
    return (y(x0+x0*1e-6)-y(x0-x0*1e-6))/(x0*2e-6)
    
def crossesZero(array):
    crosses = False
    if np.sign(array[0]/array[-1]) == -1:
        crosses = True
    return crosses
    
def zeroCrossings(x, y):
    s = np.sign(y)
    crossings = np.where(s[:-1] * s[1:] < -0.5)[0]
    return x[crossings]
    

def get_Er(data, rootChoice, zeroJrThreshold=1e-4, differentErThreshold=5.0, ErRange=99):
    assert rootChoice in ['low', 'middle', 'high']
    Jr = interp1d(data[0], data[1], kind='linear')
    Er_range = np.linspace(-ErRange, ErRange, 1000)
    Jrs = Jr(Er_range)
    zero_crossings = zeroCrossings(data[0], data[1])
    Er_candidates = [Er for Er in zero_crossings if abs(Er) <= ErRange]
    
    print("Initial ambipolar Er candidates:", Er_candidates)
    aErs = []
    aEr = None

    try:
        if len(Er_candidates) > 3:
            aEr = get_Er(data, rootChoice=rootChoice, zeroJrThreshold=zeroJrThreshold, differentErThreshold=differentErThreshold*1.01, ErRange=ErRange*0.9)
    except RecursionError:
        print("maximum recursion depth exceeded.")
        print("trying something else.")
        aErs = aErs[:1]
    
    if aEr is None:
        for er in Er_candidates:
            #print(max(er-differentErThreshold, -ErRange), min(er+differentErThreshold, ErRange))
            er_range = np.linspace(max(er-differentErThreshold, -ErRange), min(er+differentErThreshold, ErRange), 500000)
            jrs = Jr(er_range)
            if crossesZero(jrs):
                aErs.append(er_range[np.argmin(abs(jrs))])
            else:
                er_range = np.linspace(max(er-differentErThreshold/5, -ErRange), min(er+differentErThreshold/5, ErRange), 500000)
                jrs = Jr(er_range)
                if crossesZero(jrs):
                    aErs.append(er_range[np.argmin(abs(jrs))])
                else:
                    raise ValueError("can't find exact zero crossing")
            
        if len(aErs) == 1:
            if rootChoice == 'middle':
                return aErs[0]
            print("Only a single root to choose from.")
            return aErs[0]
        
        if len(aErs) == 2:
            if rootChoice == 'low' or rootChoice == 'middle': #electron root
                return aErs[0]
            return aErs[1] #ion root

        if rootChoice == 'low':
            aEr = aErs[0]
        if rootChoice == 'middle':
            aEr = aErs[1]
        if rootChoice == 'high':
            aEr = aErs[2]
            
    return aEr

def make_profile(rNs, datas, rootChoice):
    Ers = []
    for i, data in enumerate(tqdm(datas, desc="Generating Er profile...")):
        Ers.append(get_Er(data, rootChoice, ErRange=0.99999*np.max(data[0])))
        
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111)
    ax.plot(rNs, Ers, marker='o', linestyle='None', color='#012169')
    ax.set_ylabel(r"ambipolar $E_r$ (V/m)", fontsize=24)
    ax.set_xlabel(r"$\sqrt{\psi_N}$", fontsize=24)
    ax.tick_params(axis='both', labelsize=20)
    fig.show()
    return rNs, Ers
    
def make_3d_data(rNs, datas):
    Er = np.linspace(-99.9, 99.9, 30)
    Jr = []
    for i in range(0, len(datas)):
        Jr.append(interp1d(datas[i][0], datas[i][1], kind='linear')(Er))
    
    ER, RN = np.meshgrid(Er, rNs)
    JR = np.array(Jr)
    return ER, RN, JR
    
def contour(rNs, datas):
    ER, RN, JR = make_3d_data(rNs, datas)
    fig = plt.figure()
    ax = fig.add_subplot()
    plt.contourf(ER, RN, JR)
    c = plt.colorbar()
    fig.show()
    
def surface(rNs, datas):
    ER, RN, JR = make_3d_data(rNs, datas)
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(ER, RN, abs(JR))
    ax.set_zscale('log')
    fig.show()
    
#data_index = 30
#print("Flux surface:", rNs[data_index])
#print(get_Er(datas[data_index],'middle'))
rN, Er = make_profile(rNs, datas, rootChoice)

np.save("scan_info.npy", np.array([rN, np.array(Er)/1000])) # to kV/m

print("Ambipolar results")

for i in range(0, len(rN)):
    print("rN:", rN[i], "Er:", Er[i])

print("~~~ End interpret.py ~~~")
#from scipy.io import savemat

#ER, RN, JR = make_3d_data(rNs, datas)
#savemat('raderscan6.mat', {
#    'ER': ER,
#    'RN': RN,
#    'JR': JR
#})

from cProfile import label
import numpy as np
from matplotlib import pyplot as plt

def plotparallelismclose():
    times_128 = np.array([5.949620,3.376103,2.410116,1.948597,1.641621,1.430482,1.250786,1.126518,1.035110,0.917401,0.878395,0.807962,0.748122,0.753245,0.888414,0.803677,0.775998,0.772585,0.751420,0.763962,0.819055,0.719231,0.707185,0.717224,0.859470,0.695837,0.706046,0.723476])
    speedup_128 = times_128[0] / times_128
    times_256 = np.array([42.354261,22.460727,15.925813,12.393614,10.399858,8.921966,7.772290,6.905235,6.253089,5.690411,5.269419,4.870900,4.514702,4.293185,4.273316,4.069053,3.920203,3.739140,3.527814,3.436132,3.300918,3.219129,3.209854,3.051547,2.931394,2.765006,2.909860,2.804416])
    speedup_256 = times_256[0] / times_256
    times_1024 = np.array([215.781905,110.889146,78.214594,63.683998,52.827868,46.364502,41.726622,38.466800,36.931516,34.870272,35.086086,32.481311,37.471123,32.367802,28.209405,28.241629,30.512105,27.970327,21.232261,22.630575,19.267374,20.097921,17.116734,18.818783,12.443711,14.275730,15.091132,14.875479])
    speedup_1024 = times_1024[0] / times_1024
    plt.title("OMP_PROC_BIND = close")
    plt.xticks(np.arange(2, 29, 2))
    plt.yticks(np.arange(0, 20, 2))
    plt.plot(np.arange(0, 29, 1), c="grey", linestyle="--")
    plt.plot(np.arange(1, 29), np.round(speedup_128, 1), c="b", marker=".", label="128")
    plt.plot(np.arange(1, 29), np.round(speedup_256, 1), c="r", marker=".", label="256" )
    plt.plot(np.arange(1, 29), np.round(speedup_1024, 1), c="g", marker=".", label="1024" )
    plt.xlim([0, 29])
    plt.ylim([0, 18])
    plt.legend(loc="upper left")
    plt.xlabel("Number of cores")
    plt.ylabel("Speedup")
    plt.show()

def plotparallelismspread():
    times_128 = np.array([5.950223,3.648628,2.618375,2.091396,1.822410,1.593604,1.413976,1.318337,1.224516,1.153855,1.081384,1.029923,0.949118,0.950045,0.920668,0.881902,0.867521,0.846234,0.807689,0.780230,0.890790,0.840804,0.748822,0.757954,0.728700,0.749538,0.816955,0.762137])
    speedup_128 = times_128[0] / times_128
    times_256 = np.array([41.792838,23.007780,15.892794,12.462975,10.420937,8.919290,7.772728,7.139534,6.472923,6.072763,5.579524,5.127490,4.916809,4.570197,4.395612,4.057993,4.008485,3.838195,3.653942,3.525076,3.330248,3.311720,3.156242,3.098515,2.965013,2.933648,2.747209,2.760472])
    speedup_256 = times_256[0] / times_256
    times_1024 = np.array([215.899309,103.643975,74.969557,54.502181,48.344722,40.598941,34.625975,28.549383,28.190485,25.698105,23.469947,21.119348,20.428659,18.075947,17.936696,16.095167,17.723030,15.421552,17.348139,15.378277,13.315262,12.432627,12.858203,13.436907,14.445647,14.056382,13.134030,13.169946])
    speedup_1024 = times_1024[0] / times_1024
    plt.title("OMP_PROC_BIND = spread")
    plt.xticks(np.arange(2, 29, 2))
    plt.yticks(np.arange(0, 20, 2))
    plt.plot(np.arange(0, 29, 1), c="grey", linestyle="--")
    plt.plot(np.arange(1, 29), np.round(speedup_128, 1), c="b", marker=".", label="128")
    plt.plot(np.arange(1, 29), np.round(speedup_256, 1), c="r", marker=".", label="256" )
    plt.plot(np.arange(1, 29), np.round(speedup_1024, 1), c="g", marker=".", label="1024" )
    plt.xlim([0, 29])
    plt.ylim([0, 18])
    plt.legend(loc="upper left")
    plt.xlabel("Number of cores")
    plt.ylabel("Speedup")
    plt.show()

def plotvectorbar():
    op = np.array([22.2/20.117178, 176.5/161.120031, 730.6/673.476107])
    vec = np.array([22.2/5.986971, 176.5/41.834681, 730.6/213.666506])
    ind = np.arange(3)
    width = 0.3
    plt.title("Vectorisation Speedup")
    plt.bar(ind, op, width, color="b", label="Serial")
    plt.bar(ind + width, vec, width, color="g", label="Vectorised")
    plt.xticks(ind + width/2, ("128", "256", "1024"))
    plt.ylim([1, 5])
    plt.legend(loc="upper left")
    plt.xlabel("Grid size")
    plt.ylabel("Speedup")
    plt.show()

def main():
    #plotparallelismclose()
    #plotparallelismspread()
    plotvectorbar()

if __name__ == "__main__":
    main()
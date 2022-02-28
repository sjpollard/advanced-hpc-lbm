import numpy as np
from matplotlib import pyplot as plt

def plotparallelismclose():
    plt.plot(np.arange(1, 29), , c="b", marker="x", label="128")
    plt.plot(np.arange(1, 29), , c="r", marker="x", label="256" )
    plt.plot(np.arange(1, 29), , c="g", marker="x", label="1024" )
    plt.legend(loc="upper left")
    plt.xlabel("Number of cores")
    plt.ylabel("Speedup")
    plt.show()

def main():


if __name__ == "__main__":
    main()
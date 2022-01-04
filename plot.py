import matplotlib.pyplot as plt 


def plot(x, y, label, color):
    # plot density estimation from samples
    # Input
    # x          : samples
    # y          : density
    # Output
    # estDensity : estimated density of the samples
    
    plt.plot(x, y, 'r', linewidth=1.5, label=f'{label}', c=color)
    plt.grid()
    plt.legend()


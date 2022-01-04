import matplotlib.pyplot as plt 
def plot(x, y, label):
    plt.plot(x, y, 'r', linewidth=1.5, label=f'{label}')
    plt.grid()
    plt.legend()
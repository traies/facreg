import matplotlib.pyplot as plt
from utils.directory_utils import *

def plot_predict_all(path_plots, g, bsamples, subjects, name):
    plt.plot(g[0],g[1])
    plt.suptitle('Porcentaje de acierto según cantidad de autocaras',fontweight='bold')
    plt.ylabel('Acierto (%)')
    plt.xlabel('Autocaras')
    validateDirectory(path_plots)
    plt.savefig(path_plots + name + str(bsamples) + '_subjects' + str(subjects) + '.png')

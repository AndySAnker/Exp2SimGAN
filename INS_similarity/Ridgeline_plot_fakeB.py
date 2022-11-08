import pdb, sys
from numpy.random import normal
from ridgeplot import ridgeplot
import numpy as np

path = "WS_list_single_dir_fakeA/"
X_train = np.load(path+"train_WS_list_single.npy")
X_testA = np.load(path+"testA_WS_list_single.npy")
X_testB = np.load(path+"testB_WS_list_single.npy")
X_animals = np.load(path+"animals_WS_list_single.npy")
X_Exp = np.load(path+"Exp_WS_list_single.npy")
X_franken = np.load(path+"franken_WS_list_single.npy")
X_Rb2MnF4 = np.load(path+"Rb2MnF4_WS_list_single.npy")
X_Nore = np.load(path+"Nore_WS_list_single.npy")
X_TobyFit = np.load(path+"TobyFit_WS_list_single.npy")
X_Digits = np.load(path+"Digits_WS_list_single.npy")
data = [X_train, X_testB, X_Exp, X_testA, X_franken, X_Rb2MnF4, X_animals, X_Digits]

column_names = [ "Training set", "Test set incl. resolution function", "Experimental data (PCSMO)", "Test set excl. resolution function",
    "Franken data", "Simulated data (Rb2MnF4)", "Animals", "Digits"]

fig = ridgeplot(samples=data, labels=column_names) # Inspired by https://github.com/tpvasconcelos/ridgeplot

fig.update_layout(
    #title="Ridgeplot of Wasserstein distance distributions from random points in the training set",
    height=650,
    width=800,
    plot_bgcolor="rgba(255, 255, 255, 0.0)",
    xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
    yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
    xaxis_title="Wasserstein distance from random points in the training set",)

fig.show()

sys.exit()
dist_train = np.random.normal(73168.0701874221, 7803.315340491857, 1000)
dist_test = np.random.normal(75942.33640176944, 7418.772425320503, 1000)
dist_animals = np.random.normal(201397.13653693537, 9502.374979312855, 1000)
dist_Exp = np.random.normal(118876.32842441337, 5676.853831294626, 1000)
dist_franken = np.random.normal(162570.19195943078, 4892.568348443476, 1000)
dist_Rb2MnF4 = np.random.normal(153814.5484678599, 4318.1509885188, 1000)
dist_Nore = np.random.normal(125096.29854390628, 5267.589904312985, 1000)
dist_TobyFit = np.random.normal(108004.12474796743, 9276.283579072324, 1000)
dist_Digits = np.random.normal(240345.34751974104, 2727.928900462645, 1000)

data = [dist_train, dist_test, dist_TobyFit, dist_Nore, dist_Exp, dist_franken, dist_Rb2MnF4, dist_animals, dist_Digits]
fig = ridgeplot(samples=data, labels=column_names) # Inspired by https://github.com/tpvasconcelos/ridgeplot

fig.update_layout(
    #title="Ridgeplot of Wasserstein distance distributions from random points in the training set",
    height=650,
    width=800,
    plot_bgcolor="rgba(255, 255, 255, 0.0)",
    xaxis_gridcolor="rgba(0, 0, 0, 0.1)",
    yaxis_gridcolor="rgba(0, 0, 0, 0.1)",
    xaxis_title="Wasserstein distance from random points in the training set",)

fig.show()

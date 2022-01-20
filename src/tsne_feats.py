from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.manifold import TSNE
import pickle
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.font_manager import FontProperties
import pandas as pd

# TODO: Connect with other files


# df_mr_ppp = pd.read_excel("C:/Users/mrodobbe/Documents/Research/GauL-mixture_full/results_analysis.xlsx",
#                           sheet_name="Mixed Representations PPP",
#                           index_col=None)
# #
# # df_mr_pw = pd.read_excel("C:/Users/mrodobbe/Documents/Research/GauL-mixture_full/results_analysis.xlsx",
# #                          sheet_name="Mixed Features PW",
# #                          index_col=None)
#
# df_mr_mei = pd.read_excel("C:/Users/mrodobbe/Documents/Research/GauL-mixture_full/results_analysis.xlsx",
#                           sheet_name="Mixed Representations Mei",
#                           index_col=None)

with open("../mixed_representations.pickle", "rb") as f:
    mixed_representations = pickle.load(f)

df_rsq = pd.read_excel("../Figures/bp_curve.xlsx", sheet_name="First 3", index_col=None)
c_mei = df_rsq["Column13"]
df_rsq_ppp = pd.read_excel("../Figures/bp_curve.xlsx", sheet_name="Mix BP PPP", index_col=None)
c_ppp = df_rsq_ppp["Column13"]

# mr_ppp = df_mr_ppp.to_numpy()
mr = mixed_representations
# mr_pw = df_mr_pw.to_numpy()
# mr_mei = df_mr_mei.to_numpy()

# mr = np.concatenate((mr_ppp, mr_mei))
# mr = mr_mei
print(mr.shape)

# n1 = len(mr_ppp)
# n2 = len(mr_mei)

X_centered = mr - mr.mean(axis=0)
X_normalized = X_centered / sum(mr.mean(axis=0))
print(mr.mean(axis=0))

# pca = PCA(n_components=3)
# X_embedded = pca.fit_transform(all_representations)
# print(pca.explained_variance_ratio_)

# svd = TruncatedSVD(n_components=9)
# X = svd.fit_transform(X_normalized)
X = X_normalized
# X = svd.fit_transform(all_representations)
X_embedded = TSNE(n_components=2, perplexity=80, n_iter=1000).fit_transform(X)
# X_embedded = TSNE(perplexity=20, n_iter=15000).fit_transform(all_representations)

# features = []
# for i in range(len(mr)):
#     if i < n1:
#         features.append("PPP")
#     elif n1 < i < n2:
#         features.append("Mei")
#     else:
#         features.append("PW")

# train_embedded = np.loadtxt(str(file_location + "/training_coordinates.txt"))
# test_embedded = np.loadtxt(str(file_location + "/test_coordinates.txt"))
# X_embedded = np.concatenate((train_embedded, test_embedded))
# test_features = np.asarray([max(0, np.log(value)) for value in test_features])
# ln_largest_error = max(test_features)
# largest_error = int(np.floor(np.exp(ln_largest_error)))
# largest_error = max(test_features)
# ln_largest_error = np.log(largest_error)
# print(largest_error)

fig = plt.figure()
ax1 = fig.add_subplot(111)
hfont = {'fontname': 'UGent Panno Text'}
font = FontProperties(family='UGent Panno Text',
                      weight='normal',
                      style='normal', size=16)
plt.rc('font', size=18)
# h = ax1.scatter(X_embedded[:n1, 0], X_embedded[:n1, 1], c='#4a929c', edgecolors="k", s=350,
#                 alpha=0.5, label="PPP")
# t = ax1.scatter(X_embedded[n1:, 0], X_embedded[n1:, 1], c='#fad362',
#                 edgecolors="k", marker="^", s=350, alpha=0.5, label="Mei")
h = ax1.scatter(X_embedded[:, 0], X_embedded[:, 1], c=c_ppp, cmap='RdBu', edgecolors="k", s=350,
                alpha=0.5, label="PPP")
# t = ax1.scatter(X_embedded[n1:, 0], X_embedded[n1:, 1], c=c_mei, cmap='RdBu',
#                 edgecolors="k", marker="^", s=350, alpha=0.5, label="Mei")
# p = ax1.scatter(X_embedded[n1+n2:, 0], X_embedded[n1+n2:, 1], c='#d4798d',
#                 edgecolors="k", marker="*", s=450, alpha=0.5, label="PW")
# err_ax = inset_axes(ax1, width="35%", height="3%", loc=3, bbox_to_anchor=(0, 0.03, 1, 1),
#                     bbox_transform=ax1.transAxes)
# err = plt.colorbar(t, cax=err_ax, ticks=[0., np.log(10), ln_largest_error], orientation='horizontal')
# err.ax.set_xticklabels([0, 10, largest_error])
# err.ax.set_xlabel("Absolute Deviation Test Point [kJ mol$^{-1}$]", size=15, **hfont)
# err.ax.xaxis.set_ticks_position("top")
ax1.set_xlabel("t-SNE First Component", size=18, **hfont)
ax1.set_ylabel("t-SNE Second Component", size=18, **hfont)
# h.axes.get_xaxis().set_visible(False)
# h.axes.get_yaxis().set_visible(False)
# for tick in err.ax.get_xticklabels():
#     tick.set_fontname("UGent Panno Text")
# for tick in c_bar.ax.get_yticklabels():
#     tick.set_fontname("UGent Panno Text")
# plt.legend((h, t, p), ('Training molecules', 'Test molecules'), loc='lower left', bbox_to_anchor=(0, 0.15),
#            bbox_transform=ax1.transAxes, prop=font)
plt.legend()
plt.show()

# np.savetxt(str(file_location + "/training_coordinates_full.txt"), X_embedded[:n, :], fmt='%.4f')
# np.savetxt(str(file_location + "/test_coordinates_full.txt"), X_embedded[n:, :], fmt='%.4f')

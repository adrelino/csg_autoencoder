import numpy as np
import random
import pickle
import sys
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# NLDR algorithms
from sklearn.manifold import TSNE, Isomap
from sklearn.manifold import LocallyLinearEmbedding as LLE
from sklearn.decomposition import KernelPCA

NLDR = TSNE(n_components=2, verbose=True, learning_rate=500)
# NLDR = KernelPCA(n_components=2, kernel="sigmoid")
# NLDR = Isomap(n_neighbors=10, n_components=2)
# NLDR = LLE(n_neighbors=20, n_components=2, eigen_solver="dense")

class UI:
    def __init__(self, data_file_name, N):
        # Create a figure with two subplots
        self.fig, self.axes = plt.subplots(2, 2, figsize=(10, 8))
        self.fig.canvas.set_window_title("Microstructure")
        self.cid = None

        # Initialize plots
        self.axes[0, 1].set_title("Microstructure visualization")
        self.axes[1, 1].set_xlabel("Strain")
        self.axes[1, 1].set_ylabel("Stress")
        self.axes[1, 1].set_title("Stress-strain curve")

        # Read input data from pickle file
        with open(data_file_name, 'rb') as f:
            self.particles = pickle.load(f)
            self.particles = self.particles[:N]
            print('{} particles loaded'.format(N))
            random.shuffle(self.particles)

        # T-SNE
        n = (len(self.particles[0].configuration) + 1) // 2
        n_p = n * n // 2
        X = np.zeros((N, n * n + n_p * 2))
        phys_x = np.array([p.coord[0] for p in self.particles])
        phys_y = np.array([p.coord[1] for p in self.particles])
        phys_x_n = (phys_x - phys_x.min()) / (phys_x.max() - phys_x.min())
        phys_y_n = (phys_y - phys_y.min()) / (phys_y.max() - phys_y.min())
        for i, p in enumerate(self.particles):
            arr_config = p.configuration[:n, :n].flatten()
            arr_coord0 = np.full(n_p, phys_x_n[i])
            arr_coord1 = np.full(n_p, phys_y_n[i])
            X[i, :] = np.hstack((arr_config, arr_coord0, arr_coord1))
        X_embedded = NLDR.fit_transform(X)
        print(X_embedded.shape)
        for i in range(2):
            v = X_embedded[:, i]
            X_embedded[:, i] = (v - v.min()) / (v.max() - v.min())

        # Draw scatter plot of T-SNE results
        scatter = self.axes[0, 0].scatter(X_embedded[:, 0], X_embedded[:, 1])
        self.axes[0, 0].set_title("Embedded space")
        # extent = self.axes[0, 0].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        # self.fig.savefig("fig_embedded.png", bbox_inches=extent.expanded(1.2, 1.2))

        # Draw scatter plot of physical properties of microstructures
        self.axes[1, 0].scatter(phys_x, phys_y, color="g")
        self.axes[1, 0].set_xlabel("Toughness")
        self.axes[1, 0].set_ylabel("Young's Modulus")
        self.axes[1, 0].set_title("Gamut")
        extent = self.axes[1, 0].get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        self.fig.savefig("fig_gamut.png", bbox_inches=extent.expanded(1.4, 1.4))

        # Setup annotation of the scatter plot
        annot = self.axes[0, 0].annotate("", xy=(0, 0), xytext=(20, 20),
                                     textcoords="offset points",
                                     bbox=dict(boxstyle="round", fc="w"),
                                     arrowprops=dict(arrowstyle="->"))
        annot.set_visible(False)

        # Update annotation with image of microstructure
        def update_annot(ind):
            pos = scatter.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            annot.get_bbox_patch().set_alpha(0.4)
            p = self.particles[ind["ind"][0]]

            # Draw image of current microstructure
            res = p.configuration.shape
            self.axes[0, 1].clear()
            self.axes[0, 1].set_title("Microstructure visualization")
            for i in range(res[0]):
                for j in range(res[1]):
                    nodes = [(i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)]
                    scale = 1. / res[0]
                    nodes = [(n[0] * scale, n[1] * scale) for n in nodes]

                    if p.configuration[i, j] == 0:
                        color = (0.2, 0.2, 0.2)
                    elif p.configuration[i, j] == 1:
                        color = (1.0, 0.5, 0.7)
                    else:
                        color = (1, 1, 1)

                    t1 = plt.Polygon(nodes, color=color)
                    self.axes[0, 1].add_patch(t1)

            # Draw graph of physical property
            colors = ["g" for i in range(N)] + ["r"]
            xs = np.hstack((phys_x, [p.coord[0]]))
            ys = np.hstack((phys_y, [p.coord[1]]))
            self.axes[1, 0].clear()
            self.axes[1, 0].scatter(xs, ys, color=colors)
            self.axes[1, 0].set_xlabel("Toughness")
            self.axes[1, 0].set_ylabel("Young's Modulus")
            self.axes[1, 0].set_title("Gamut")

            # Draw stress-strain curve
            self.axes[1, 1].clear()
            self.axes[1, 1].set_xlabel("Strain")
            self.axes[1, 1].set_ylabel("Stress")
            self.axes[1, 1].set_title("Stress-strain curve")
            self.axes[1, 1].plot([0.0] + p.curve[0], [0.0] + p.curve[1])

        # Define mouse event
        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == self.axes[0, 0]:
                cont, ind = scatter.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    self.fig.canvas.draw_idle()
                elif vis:
                    annot.set_visible(False)
                    self.fig.canvas.draw_idle()

        # Register mouse event
        if self.cid is not None:
            self.fig.canvas.mpl_disconnect(self.cid)
        self.cid = self.fig.canvas.mpl_connect("motion_notify_event", hover)

        # Show the figure
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    ui = UI()

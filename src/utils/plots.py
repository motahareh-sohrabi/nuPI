import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import ListedColormap
from sklearn import svm


def plot_hyperplane(w, b, X, plot_paralles=False, color="k"):
    xx = np.linspace(X[:, 0].min(), X[:, 0].max())

    yy = (-w[0] / w[1]) * xx - (b / w[1])
    plt.plot(xx, yy, "-", color=color, alpha=0.5)

    if plot_paralles:
        for sign in [-1, 1]:
            plane = yy + sign * (1 / w[1])
            plt.plot(xx, plane, "--", color=color, alpha=0.5)


def plot_max_margin(X, y, plot_parallels=False):
    clf = svm.SVC(kernel="linear", C=1e8)
    clf.fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    plot_hyperplane(w, b, X, plot_paralles=True, color="green")


def plot_2d_decision_boundary(
    model,
    X,
    y,
    example_size=None,
    contour_levels=[],
    up_title="",
    low_title="",
    save_path=None,
    show=False,
    filename_prefix="",
    show_max_margin=False,
):
    """
    Plots the decision boundary of a 2D model for a binary classification problem.
    """

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    samples_per_dim = 150
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    x_min = y_min = min(x_min, y_min)
    x_max = y_max = max(x_max, y_max)

    x_linspace = torch.linspace(x_min, x_max, samples_per_dim)
    y_linspace = torch.linspace(y_min, y_max, samples_per_dim)
    xx, yy = torch.meshgrid(x_linspace, y_linspace, indexing="ij")
    Z = model(torch.cat((xx.reshape(-1, 1), yy.reshape(-1, 1)), dim=1).to(DEVICE))

    # Keep probability for class 1
    Z = torch.max(Z, dim=1)[0].detach().cpu().numpy()

    Z = Z.reshape(xx.shape)
    cmap = ListedColormap(["#FF0000", "#0000FF"])

    fig, ax = plt.subplots()
    cf = ax.contourf(xx, yy, Z, cmap="RdBu", alpha=0.2)

    if show_max_margin:
        plot_max_margin(X, y, plot_parallels=True)

    # Always show contour levels at and around the decision boundary
    base_contour_levels = [-1.0, 0.0, 1.0]
    contour = ax.contour(
        xx, yy, Z, levels=base_contour_levels, colors="black", alpha=0.4, linestyles="dashed", linewidths=0.5
    )
    plt.clabel(contour, inline=True, fontsize=6)

    # Show any extra contour levels requested
    custom_contour_levels = sorted(list(set(contour_levels) - set(base_contour_levels)))
    if len(custom_contour_levels) > 0:
        contour = ax.contour(
            xx, yy, Z, levels=custom_contour_levels, colors="black", alpha=0.4, linestyles="dashed", linewidths=1
        )
        plt.clabel(contour, inline=True, fontsize=7)

    # Show provided datapoints color-coded by class
    plt.scatter(X[:, 0], X[:, 1], marker="+", c=y, cmap=cmap, s=10, alpha=0.35)

    if example_size is not None:
        min_size, max_size = example_size.min() + 1e-8, example_size.max()
        if max_size == 0:
            low_title += " (**no SVs**)"
        else:
            example_size = 100 * (example_size - min_size) / (max_size - min_size)
            plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, s=example_size, alpha=0.5, edgecolors=(0, 0, 0, 0.5))

    plt.suptitle(up_title, fontsize=11)
    plt.title(low_title, fontsize=9)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    ax = plt.gca()
    ax.set_aspect("equal", adjustable="box")

    if save_path is not None:
        file_name = filename_prefix + f"decision_boundary_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.pdf"
        plt.savefig(save_path + file_name, dpi=600)

    if show:
        plt.show()

    return fig

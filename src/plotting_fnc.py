import matplotlib.pyplot as plt
import numpy as np

from utils.frenet_cartesian_converter import convert_frenet_to_cartesian


def plot_res(spline, simX, simU, realX, save_figures: bool = True, filename: str = ""):
    # plot results
    t = np.linspace(0, len(simU), len(simU))
    figure = plt.figure()
    plt.subplot(2, 1, 1)
    plt.step(t, simU[:, 0], color="r")
    plt.step(t, simU[:, 1], color="g")
    plt.title("Result plots")
    plt.legend(["dD", "ddelta"], bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)
    plt.ylabel("u")
    plt.xlabel("iteration")
    plt.grid(True)
    plt.subplot(2, 1, 2)
    plt.plot(t, simX[:, 1:])
    plt.plot(t, realX[:, 1:])
    plt.ylabel("x")
    plt.xlabel("iteration")
    plt.legend(
        [
            "n",
            "alpha",
            "v",
            "D",
            "delta",
            "real_n",
            "real_alpha",
            "real_v",
            "real_D",
            "real_delta",
        ],
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        borderaxespad=0.0,
    )
    # figure.set_size_inches(1.8, 0.9)
    plt.grid(True)
    if save_figures:
        plt.savefig(f"{filename}_state_change.png")

    plt.show()
    figure = plt.figure()
    plt.plot(t, simX[:, 0])
    plt.show()
    figure = plt.figure()
    sd = np.array(simX[:, :2])
    xy = convert_frenet_to_cartesian(spline, sd)
    xy_desired = np.zeros((sd.shape))
    for i in range(sd.shape[0]):
        xy_desired[i, :] = spline.get_coordinate(sd[i, 0])
    np.save(f"{filename}_path_data.npy", xy)
    np.save(f"{filename}_desired_path_data.npy", xy_desired)
    plt.plot(-xy[:, 1], xy[:, 0])
    plt.plot(-xy_desired[:, 1], xy_desired[:, 0])
    plt.legend(["real path", "desired path"])
    if save_figures:
        plt.savefig(f"{filename}_path.png")
    plt.show()

#####
# title: run_physigym_tib_episodes.py
#
# language: python3
# library: gymnasium, numpy,
#   and the extending and physigym custom_modules
#
# date: 2024-spring
# license: bsb-3-clause
# author: Alexandre Bertin, Elmar Bucher
# input: https://gymnasium.farama.org/main/
# original source code: https://github.com/Dante-Berth/PhysiGym
#
# run:
#   1. copy this file into the PhysiCell root folder
#   2. python3 run_physigym_tib_episodes.py
#
# description:
#   python script to run multiple episodes from the physigym tib model.
#####


# library
import argparse
from extending import physicell
import gymnasium
import numpy as np
import physigym
from random import randrange


def _plot_obs(o_observation, list_name_obs):
    import matplotlib.pyplot as plt
    import numpy as np

    # Assume o_observation is your tensor of shape (4, 64, 64)
    # Example: o_observation = np.random.rand(4, 64, 64)

    n_channels = o_observation.shape[0]

    fig, axes = plt.subplots(1, n_channels, figsize=(4 * n_channels, 4))
    for i in range(n_channels):
        ax = axes[i]
        im = ax.imshow(o_observation[i], cmap="grey", vmin=0, vmax=255)
        ax.set_title(f"Channel")
        ax.axis("off")
        ax.set_title(f"Channel: {list_name_obs[i][0].upper() + list_name_obs[i][1:]}")
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.subplots_adjust(top=0.85)
    plt.tight_layout()
    plt.show()


# function
def run(
    s_settingxml="config/PhysiCell_settings.xml",
    r_maxtime=1440.0,
    i_thread=8,
    i_seed=3,
):
    # load PhysiCell Gymnasium environment
    # %matplotlib
    # env = gymnasium.make('physigym/ModelPhysiCellEnv-v0', settingxml='config/PhysiCell_settings.xml', figsize=(8,6), render_mode='human', render_fps=10)
    env = gymnasium.make(
        "physigym/ModelPhysiCellEnv-v0",
        settingxml=s_settingxml,
        observation_mode="delaunay_graph",
    )

    # episode loop
    for i_episode in range(3):
        # manipulate setting xml before reset
        env.get_wrapper_attr("x_root").xpath("//overall/max_time")[0].text = str(
            r_maxtime
        )
        env.get_wrapper_attr("x_root").xpath("//parallel/omp_num_threads")[
            0
        ].text = str(i_thread)
        env.get_wrapper_attr("x_root").xpath("//save/folder")[
            0
        ].text = f"output/episode{str(i_episode).zfill(8)}"

        # reset the environment
        r_reward = 0.0
        o_observation, d_info = env.reset()

        # time step loop
        b_episode_over = False
        while not b_episode_over:
            # policy according to o_observation
            d_observation = o_observation
            d_action = {"drug_1": np.array([0.0], dtype=np.float16)}
            print(f"Reward:{r_reward}")
            print(f"Info: {d_info}")
            # action
            o_observation, r_reward, b_terminated, b_truncated, d_info = env.step(
                d_action
            )
            # _plot_obs(o_observation, list_name_obs=env.unwrapped.substrate_unique)
            b_episode_over = b_terminated or b_truncated
            # print(b_episode_over)
    # drop the environment
    env.close()


# run
if __name__ == "__main__":
    print(f"run physigym episodes ...")

    # argv
    parser = argparse.ArgumentParser(
        prog=f"run physigym episodes",
        description=f"script to run physigym episodes.",
    )
    # settingxml file
    parser.add_argument(
        "settingxml",
        # type = str,
        nargs="?",
        default="config/PhysiCell_settings.xml",
        help="path/to/settings.xml file.",
    )
    # max_time
    parser.add_argument(
        "-m",
        "--max_time",
        type=float,
        nargs="?",
        default=1440.0,
        help="set overall max_time in min in the settings.xml file.",
    )
    # thread
    parser.add_argument(
        "-t",
        "--thread",
        type=int,
        nargs="?",
        default=8,
        help="set parallel omp_num_threads in the settings.xml file.",
    )
    # seed
    parser.add_argument(
        "-s",
        "--seed",
        # type = int,
        nargs="?",
        default="none",
        help="set options random_seed in the settings.xml file and python.",
    )

    # parse arguments
    args = parser.parse_args()
    # print(args)

    # processing
    run(
        s_settingxml=args.settingxml,
        i_thread=args.thread,
        i_seed=None if args.seed.lower() == "none" else int(args.seed),
    )

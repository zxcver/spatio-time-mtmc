
import argparse

from evaluation.evalreid import run


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_npy", default="/home/lhao/workspace/electricity-mtmc/exp/distmatrix.npz", 
                            type=str, help="file of distmat, q_pids, g_pids, q_camids, g_camids")
    opt = parser.parse_args()
    run(opt)
import os
import argparse

parser = argparse.ArgumentParser(description="Run attention2vec generation embeddings suite.")
parser.add_argument('--dataset', nargs='?', default='cora',
                    help='Input graph name for saving files')
args = parser.parse_args()


dataset = args.dataset
gamma = [10]
R = [0.5]
T = [3, 4]
train = [20, 30]

for g in gamma:
    for r in R:
        for t in T:
            for tr in train:
                print("----------------------------")
                print("Parameters : ", g, r, t, tr)
                print("----------------------------")
                cmd = "python main.py --dataset {} --attn2v_iter {} --r {} --t {} --train_per {}".format(dataset, g, r, t, tr)
                print(cmd + "\n")
                os.system(cmd)

print("Done!")

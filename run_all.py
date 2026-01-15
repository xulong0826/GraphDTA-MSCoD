import subprocess

def main():
    # where the first argument is for the index of the datasets, 0/1 for 'davis' or 'kiba', respectively;
    # the second argument is for the index of the models, 0/1/2/3 for GINConvNet, GATNet, GAT_GCN, or GCNNet, respectively;
    # the third argument is for the CUDA device, 0/1 for 'cuda:0' or 'cuda:1', respectively;
    # cmds = [
    #     "python training_validation.py 1 0 0",
    #     "python training_validation.py 1 1 0",
    #     "python training_validation.py 1 2 0",
    #     "python training_validation.py 1 3 0",
    #     "python training_validation.py 1 0 0",
    #     "python training_validation.py 1 1 0",
    #     "python training_validation.py 1 2 0",
    #     "python training_validation.py 1 3 0",
    #     "python training_validation.py 1 0 0",
    #     "python training_validation.py 1 1 0",
    #     "python training_validation.py 1 2 0",
    #     "python training_validation.py 1 3 0"
    #     ]
    cmds = [
    "python training_validation.py 1 2 0",
    "python training_validation.py 1 3 0"
    ]

    for cmd in cmds:
        print(f"🚀 Running: {cmd}")
        subprocess.run(cmd, shell=True)

if __name__ == "__main__":
    main()

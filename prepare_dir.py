"Copy the checkpoints of the given model according to the desired format."
import argparse
import os
import shutil


def parse_arguments():
    parser = argparse.ArgumentParser(description="Format the checkpoints of the given model")
    parser.add_argument(
        "--path",
        type=str,
        help="Path of the model.",
        required=True,
    )
 
    args = parser.parse_args()
    # Sanity checks
    if "trained_models" not in args.path:
        raise ValueError("Give a path of a trained model.")  

    return args


def main()
    args = parse_arguments()

    dir_content = os.listdir(args.path)
    steps = [item.split("_")[1] for item in dir_content if "step" in item]

    for step in steps:
        shutil.copy2(f"{args.path}/step_{step}/pytorch_model.bin",f"{args.path}/model_{step}.pth")


if __name__ == '__main__':
    main()

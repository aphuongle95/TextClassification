import argparse
from train import train
from infer import infer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_search', action='store_true', default=False, help="if run training with random search on hyperparameters")
    parser.add_argument('--model_name', help="add model name, example: vinai/xphonebert-base or FPTAI/vibert-base-cased")
    parser.add_argument('-lr', '--learning_rate', default=2e-5)
    parser.add_argument('--weight_decay', default=0.1)
    parser.add_argument("--num_epochs", default=5)
    parser.add_argument("--infer", default=False, action='store_true' )
    args = parser.parse_args()

    if not args.infer:
        train(model_name = args.model_name, random_search = args.random_search)
    else:
        infer(model_name = args.model_name)

if __name__ == "__main__":
    main()
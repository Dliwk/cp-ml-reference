def run(args):
    import pipeline2

    pipeline2.run(data_zip_path=args.data, model_path=args.model, use_pretrained=args.fetch_pretrained)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="main.py", description="train a model", epilog="use at your own risk"
    )

    parser.add_argument("--data", help="path to train.zip", required=True)
    parser.add_argument("--model", help="path to resulting model.pt", required=True)
    parser.add_argument("--fetch-pretrained", help="whether to fetch pretrained model weights", action='store_true')

    args = parser.parse_args()

    run(args)

def run(args):
    import pipeline2
    import pipeline3

    pipeline2.run(
        data_zip_path=args.data,
        model_path=args.model,
        use_pretrained=args.fetch_pretrained,
        test_path=args.test_path,
    )

    # Создаст submission.csv
    pipeline3.run(test_path=args.test_path)

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        prog="main.py", description="train a model", epilog="use at your own risk"
    )

    parser.add_argument("--data", help="path to train.zip", required=True)
    parser.add_argument("--model", help="path to model.pt", required=True)
    parser.add_argument(
        "--fetch-pretrained",
        help="whether to fetch pretrained model weights",
        action="store_true",
    )
    parser.add_argument("--test-path", help="path to directory with TEST_RES.csv", required=True)

    args = parser.parse_args()

    run(args)

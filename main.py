import argparse

from src.train import train, download_text8
from src.evaluate import run_evaluation

def main():
    parser = argparse.ArgumentParser(description="Main script.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser_train = subparsers.add_parser("train", help="Train Word2Vec model")
    parser_train.add_argument("--data", type=str, default=None)
    parser_train.add_argument("--max_chars", type=int, default=None)
    parser_train.add_argument("--epochs", type=int, default=5)
    parser_train.add_argument("--embedding_dim", type=int, default=100)
    parser_train.add_argument("--window_size", type=int, default=5)
    parser_train.add_argument("--n_negatives", type=int, default=5)
    parser_train.add_argument("--batch_size", type=int, default=512)
    parser_train.add_argument("--lr", type=float, default=0.025, dest="learning_rate")
    parser_train.add_argument("--min_lr", type=float, default=0.0001)
    parser_train.add_argument("--min_count", type=int, default=5)
    parser_train.add_argument("--subsample_threshold", type=float, default=1e-4)
    parser_train.add_argument("--save", type=str, default="./embeddings/model.npz", dest="save_path")

    parser_eval = subparsers.add_parser("evaluate", help="Ewaluuj wytrenowany model")
    parser_eval.add_argument("--model_path", type=str, default="./embeddings/model_epoch5.npz")

    args = parser.parse_args()

    if args.command == "train":
        data_path = args.data if args.data else download_text8(save_dir="./data")

        with open(data_path, "r", encoding="utf-8") as fh:
            text = fh.read(args.max_chars) if args.max_chars else fh.read()

        train(
            text=text,
            epochs=args.epochs,
            embedding_dim=args.embedding_dim,
            window_size=args.window_size,
            n_negatives=args.n_negatives,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            min_count=args.min_count,
            subsample_threshold=args.subsample_threshold,
            save_path=args.save_path,
        )

    elif args.command == "evaluate":
        run_evaluation(args.model_path)

if __name__ == "__main__":
    main()
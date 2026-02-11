from __future__ import annotations

import argparse
from pathlib import Path

from .indexer import build_index
from .search import semantic_search


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="browser-brain", description="Semantic Search по истории браузера")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Построить индекс по истории браузера")
    p_index.add_argument("--browser", choices=["chrome", "firefox"], required=True)
    p_index.add_argument("--history-path", type=Path, default=None)
    p_index.add_argument("--output-dir", type=Path, default=Path(".index"))
    p_index.add_argument("--limit", type=int, default=None)
    p_index.add_argument("--chunk-size", type=int, default=260)
    p_index.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")

    p_search = sub.add_parser("search", help="Семантический поиск по индексу")
    p_search.add_argument("--index-dir", type=Path, default=Path(".index"))
    p_search.add_argument("--query", type=str, required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--model-name", type=str, default=None)

    return parser


def cmd_index(args: argparse.Namespace) -> int:
    stats = build_index(
        browser=args.browser,
        output_dir=args.output_dir,
        history_path=args.history_path,
        limit=args.limit,
        chunk_size=args.chunk_size,
        model_name=args.model_name,
    )
    print(f"Indexed records={stats['records']}, chunks={stats['chunks']}, dim={stats['embedding_dim']}")
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    results = semantic_search(
        query=args.query,
        index_dir=args.index_dir,
        top_k=args.top_k,
        model_name=args.model_name,
    )
    if not results:
        print("No results.")
        return 0

    for item in results:
        print(f"{item['rank']}. [{item['score']:.4f}] {item['url']}")
        print(f"   title: {item['title']}")
        print(f"   when : {item['visited_at']} ({item['browser']})")
        print(f"   text : {item['text'][:200]}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "index":
        return cmd_index(args)
    if args.command == "search":
        return cmd_search(args)

    parser.error("Unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

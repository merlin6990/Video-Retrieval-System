import argparse
import json
import os
import torch
import faiss
from transformers import CLIPProcessor, CLIPModel
from src.pipeline import (
    initialize_database,
    add_to_db,
    retrieve
)

def load_config(config_path="config.json"):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file {config_path} not found.")
    with open(config_path, "r") as f:
        return json.load(f)

def load_clip_model(model_name, device):
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)
    print("Model loaded.")
    return model, processor

def ensure_index(index_path, embedding_dim):
    if os.path.exists(index_path):
        print("Loading existing FAISS index...")
        index = faiss.read_index(index_path)
    else:
        print("Initializing new FAISS index...")
        index = initialize_database(embedding_dim, index_path)
    return index

def main():
    parser = argparse.ArgumentParser(description="Video Retrieval System")
    parser.add_argument("--config", type=str, default="config.json", help="Path to config file")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Add videos
    add_parser = subparsers.add_parser("add", help="Add videos to the database")
    add_parser.add_argument("--video_paths", nargs="+", required=True, help="Paths to video files")
    add_parser.add_argument("--frames_per_second", type=int, default=1, help="Frame sampling rate (frames per second)")

    # Retrieve videos
    retrieve_parser = subparsers.add_parser("retrieve", help="Retrieve similar videos by text prompt")
    retrieve_parser.add_argument("--query", type=str, required=True, help="Text prompt (English or Persian)")
    retrieve_parser.add_argument("--top_k", type=int, default=3, help="Number of top results to retrieve")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and index
    model, processor = load_clip_model(config["clip_model_name"], device)
    index = ensure_index(config["index_path"], config["embedding_dim"])

    if args.command == "add":
        add_to_db(
            video_paths=args.video_paths,
            index=index,
            index_path=config["index_path"],
            model=model,
            processor=processor,
            mapping_path=config["mapping_path"],
            frames_per_second=args.frames_per_second
        )

    elif args.command == "retrieve":
        results = retrieve(
            query=args.query,
            model=model,
            processor=processor,
            index=index,
            top_k=args.top_k,
            mapping_path=config["mapping_path"]
        )

        print("\nTop matching results:")
        for i, res in enumerate(results):
            if res:
                print(f"{i+1}. Video Path: {res['path']}, Time (s): {res['time']}")
            else:
                print(f"{i+1}. No metadata found for this result.")

if __name__ == "__main__":
    main()

from PIL import Image
import torch
import cv2
import os
import json
import faiss
import numpy as np

def extract_frames(video_path, sampling_rate=1):

    # Check if the video file exists
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"The video file {video_path} does not exist.")

    # Initialize video capture
    video_capture = cv2.VideoCapture(video_path)

    if not video_capture.isOpened():
        raise ValueError(f"Cannot open video file {video_path}.")

    frames = []
    frames_metadata = []
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)  # Frames per second of the video
    total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    # Calculate the interval in terms of frame indices
    interval = int(frame_rate / sampling_rate)

    frame_index = 0
    time = 0
    while frame_index < total_frames:
        video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = video_capture.read()

        if not success:
            break

        frames.append(frame)

        metadata = {'path':video_path, 'time':time}
        frames_metadata.append(metadata)

        frame_index += interval
        time += 1

    video_capture.release()
    return frames, frames_metadata


def convert_frames_to_vectors(frames, model, processor):

    # Convert frames to PIL images
    pil_images = [Image.fromarray(frame) for frame in frames]

    # Preprocess images for CLIP
    inputs = processor(images=pil_images, return_tensors="pt", padding=True)

    # Use the provided CLIP model to get image embeddings
    with torch.no_grad():
        embeddings = model.get_image_features(**inputs)

    # Normalize embeddings for cosine similarity
    normalized_embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

    # Convert embeddings to a list of numpy arrays
    vector_list = normalized_embeddings.cpu().numpy()

    return vector_list


def add_embeddings_with_mapping(index, index_path, new_embeddings, metadata_list, mapping_path="vector_mapping.json"):

    # Check the dimensionality
    d = index.d  # Dimensionality of the index
    if new_embeddings.shape[1] != d:
        raise ValueError(f"Embedding dimension {new_embeddings.shape[1]} does not match index dimension {d}.")

    # Load or initialize the mapping
    try:
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
    except FileNotFoundError:
        mapping = {}

    # Get the current number of vectors in the index
    current_count = index.ntotal

    # Add new embeddings to the FAISS index
    index.add(new_embeddings)
    print(f"Added {new_embeddings.shape[0]} new embeddings to the FAISS index.")

    # Update the mapping with new indices
    for i, metadata in enumerate(metadata_list):
        mapping[current_count + i] = metadata

    # Save the updated mapping to disk
    with open(mapping_path, "w") as f:
        json.dump(mapping, f)
    print(f"Updated mapping saved at {mapping_path}.")

    # Save the updated FAISS index
    faiss.write_index(index, index_path)
    print(f"Updated FAISS index saved at {index_path}.")

    return index


def retrieve_similar_images_with_text(query, model, processor, index, top_k=3):

    # Preprocess the text query
    inputs = processor(text=query, return_tensors="pt", padding=True)

    # Extract text embeddings using CLIP
    with torch.no_grad():
        query_features = model.get_text_features(**inputs)

    # Normalize the text embeddings for cosine similarity (if index uses cosine)
    query_features = torch.nn.functional.normalize(query_features, p=2, dim=1)
    query_features = query_features.cpu().numpy().astype(np.float32)

    # Search for similar images in the FAISS index
    distances, indices = index.search(query_features, top_k)

    return query, distances, indices


def get_metadata_from_faiss_indices(indices, mapping_path="vector_mapping.json"):

    # Load the mapping
    with open(mapping_path, "r") as f:
        mapping = json.load(f)
    
    # Retrieve metadata for each index
    metadata = [mapping.get(str(index), None) for index in indices[0]]
    return metadata


def initialize_database(dimension, index_path):  
  index = faiss.IndexFlatIP(dimension)

  # Save the FAISS index to disk
  faiss.write_index(index, index_path)
  print(f"FAISS index saved at {index_path}.")
  return index


def add_to_db(video_paths, index, index_path, model, processor, mapping_path="vector_mapping.json", sampling_rate=1):

  for path in video_paths:
    frames, metadata_list = extract_frames(path, sampling_rate)
    embeddings = convert_frames_to_vectors(frames, model, processor)
    add_embeddings_with_mapping(index, index_path, embeddings, metadata_list, mapping_path=mapping_path)

  print(f"videos with following paths have been added to the database:")
  for path in video_paths:
    print(path)


def retrieve(query, model, processor, index, top_k=3, mapping_path="vector_mapping.json"):

  _, _, indices = retrieve_similar_images_with_text(query, model, processor, index, top_k=top_k)  
  metadata = get_metadata_from_faiss_indices(indices, mapping_path=mapping_path)
  return metadata

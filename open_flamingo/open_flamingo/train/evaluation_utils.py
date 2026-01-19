# import tensorflow as tf
# import tensorflow_hub as hub
# import numpy as np

from sentence_transformers import SentenceTransformer, util
import torch





# # Load Universal Sentence Encoder
# embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
# def semantic_similarity(reference_texts, generated_texts):
#     # Generate embeddings
#     reference_embeddings = embed(reference_texts)
#     generated_embeddings = embed(generated_texts)
    
#     # Compute similarity: Use cosine similarity
#     similarity_matrix = np.inner(reference_embeddings, generated_embeddings)
    
#     # Diagonal elements give the similarity scores between corresponding pairs
#     similarity_scores = np.diag(similarity_matrix)
#     return similarity_scores



# Load a pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')
def sentence_transformers_similarity(reference_texts, generated_texts):
    # Generate embeddings
    reference_embeddings = model.encode(reference_texts, convert_to_tensor=True)
    generated_embeddings = model.encode(generated_texts, convert_to_tensor=True)
    
    # Compute cosine similarities
    cosine_scores = util.pytorch_cos_sim(reference_embeddings, generated_embeddings)

    # Extract diagonal elements for corresponding text pairs
    similarity_scores = [cosine_scores[i][i].item() for i in range(len(reference_texts))]
    return similarity_scores

# # Example usage
# reference_texts = ["This is a test.", "Here is another example."]
# generated_texts = ["This is a trial.", "Here is a different sample."]
# scores = sentence_transformers_similarity(reference_texts, generated_texts)

# print("Similarity Scores:", scores)

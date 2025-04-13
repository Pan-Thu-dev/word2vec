import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from transformers import BertTokenizer, BertModel
import torch
import logging
import uvicorn

# --- Configuration ---
MODEL_NAME = 'bert-base-uncased'
DIMENSIONS = 3
PERPLEXITY = 5
GLOBE_RADIUS = 10
MIN_CLUSTERS = 5  # Minimum clusters for ≥10 words
MAX_CLUSTERS = 20  # Cap for large inputs
CLUSTER_DISTANCE_THRESHOLD = 0.7  # Block unrelated clusters
WITHIN_CLUSTER_SIMILARITY = 0.8  # Green links
CROSS_CLUSTER_SIMILARITY_MIN = 0.5  # Orange links
CROSS_CLUSTER_SIMILARITY_MAX = 0.75
MIN_SIMILARITY = 0.1  # Avoid near-zero links

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI()

# --- CORS ---
origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Load Model ---
tokenizer = None
model = None

@app.on_event("startup")
async def load_model():
    global tokenizer, model
    logger.info(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertModel.from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# --- Helper Function ---
def process_words(words: list[str]):
    if not words:
        return {"words": [], "positions": [], "links": [], "clusters": []}
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    logger.info(f"Processing words: {words}")
    # 1. Get Embeddings
    word_vectors = []
    valid_words = []
    for word in words:
        try:
            inputs = tokenizer(word, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = model(**inputs)
            embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()
            word_vectors.append(embedding)
            valid_words.append(word)
        except Exception as e:
            logger.warning(f"Could not get embedding for word '{word}': {e}")
            continue

    if not valid_words:
        return {"words": [], "positions": [], "links": [], "clusters": []}

    word_vectors = np.array(word_vectors)

    # Handle single word case
    if len(valid_words) == 1:
        return {"words": valid_words, "positions": [[0, 0, 0]], "links": [], "clusters": [0]}

    # 2. Dimensionality Reduction (t-SNE)
    current_perplexity = min(PERPLEXITY, len(valid_words) - 1)
    if current_perplexity <= 0:
        logger.warning("Not enough samples for t-SNE.")
        return {"words": valid_words, "positions": [], "links": [], "clusters": []}

    tsne = TSNE(n_components=DIMENSIONS, random_state=42, perplexity=current_perplexity, max_iter=300, init='pca')
    try:
        word_vectors_3d = tsne.fit_transform(word_vectors)
    except ValueError as e:
        logger.error(f"t-SNE failed: {e}. Word vectors shape: {word_vectors.shape}")
        return {"words": valid_words, "positions": [], "links": [], "clusters": []}

    # 3. Map to Sphere
    norms = np.linalg.norm(word_vectors_3d, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8
    word_vectors_sphere = word_vectors_3d / norms * GLOBE_RADIUS

    # 4. Clustering with AgglomerativeClustering
    if len(valid_words) > 1:
        n_clusters = min(MAX_CLUSTERS, max(MIN_CLUSTERS, len(valid_words) // 20))
        if len(valid_words) < 10:
            n_clusters = max(1, len(valid_words) // 2)
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        try:
            cluster_labels = clusterer.fit_predict(word_vectors)
        except Exception as e:
            logger.error(f"AgglomerativeClustering failed: {e}. Using single cluster.")
            cluster_labels = np.zeros(len(valid_words), dtype=int)
    else:
        cluster_labels = np.zeros(1, dtype=int)

    unique_clusters = np.unique(cluster_labels)

    # 5. Calculate Cluster Centroids
    centroids = []
    for cluster in unique_clusters:
        cluster_vectors = word_vectors[cluster_labels == cluster]
        centroid = np.mean(cluster_vectors, axis=0) if len(cluster_vectors) > 0 else np.zeros(word_vectors.shape[1])
        centroids.append(centroid)
    centroids = np.array(centroids)

    # 6. Calculate Similarities & Links
    links = []
    if len(valid_words) > 1:
        similarity_matrix = cosine_similarity(word_vectors)
        centroid_similarity = cosine_similarity(centroids) if len(unique_clusters) > 1 else np.ones((1, 1))

        for i in range(len(valid_words)):
            for j in range(i + 1, len(valid_words)):
                similarity = similarity_matrix[i, j]
                if similarity < MIN_SIMILARITY:
                    logger.info(f"No link between '{valid_words[i]}' and '{valid_words[j]}' (similarity {similarity:.2f} < {MIN_SIMILARITY})")
                    continue

                cluster_i = cluster_labels[i]
                cluster_j = cluster_labels[j]

                # Within-cluster links (green)
                if cluster_i == cluster_j:
                    if similarity >= WITHIN_CLUSTER_SIMILARITY:
                        links.append({
                            "source": i,
                            "target": j,
                            "similarity": float(similarity)
                        })
                        logger.info(f"Created link between '{valid_words[i]}' and '{valid_words[j]}' (same cluster {cluster_i}, similarity {similarity:.2f})")
                    else:
                        logger.info(f"No link between '{valid_words[i]}' and '{valid_words[j]}' (same cluster {cluster_i}, similarity {similarity:.2f} < {WITHIN_CLUSTER_SIMILARITY})")
                    continue

                # Cross-cluster links (orange)
                if similarity >= CROSS_CLUSTER_SIMILARITY_MIN and similarity <= CROSS_CLUSTER_SIMILARITY_MAX:
                    cluster_sim = centroid_similarity[cluster_i, cluster_j] if cluster_i < len(centroids) and cluster_j < len(centroids) else 0
                    if cluster_sim >= CLUSTER_DISTANCE_THRESHOLD:
                        links.append({
                            "source": i,
                            "target": j,
                            "similarity": float(similarity)
                        })
                        logger.info(f"Created link between '{valid_words[i]}' and '{valid_words[j]}' (cross-cluster {cluster_i} vs {cluster_j}, similarity {similarity:.2f}, cluster similarity {cluster_sim:.2f})")
                    else:
                        logger.info(f"No link between '{valid_words[i]}' and '{valid_words[j]}' (cross-cluster {cluster_i} vs {cluster_j}, low cluster similarity {cluster_sim:.2f} < {CLUSTER_DISTANCE_THRESHOLD})")
                else:
                    logger.info(f"No link between '{valid_words[i]}' and '{valid_words[j]}' (cross-cluster {cluster_i} vs {cluster_j}, similarity {similarity:.2f} not in {CROSS_CLUSTER_SIMILARITY_MIN}–{CROSS_CLUSTER_SIMILARITY_MAX})")

    logger.info(f"Processing complete. Found {len(links)} links across {len(unique_clusters)} clusters.")
    return {
        "words": valid_words,
        "positions": word_vectors_sphere.tolist(),
        "links": links,
        "clusters": cluster_labels.tolist()
    }

# --- API Endpoint ---
@app.get("/api/word-data")
async def get_word_data(words_query: str):
    words_list = [word.strip() for word in words_query.split(',') if word.strip()]
    if not words_list:
        return {"words": [], "positions": [], "links": [], "clusters": []}
    try:
        data = process_words(words_list)
        return data
    except Exception as e:
        logger.error(f"Error processing words in endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during word processing: {e}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
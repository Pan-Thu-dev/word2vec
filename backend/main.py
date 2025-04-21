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
import redis
import pickle
from pydantic import BaseModel
from typing import List
from dotenv import load_dotenv
import random  

# Load environment variables
load_dotenv()

# --- Configuration ---
MODEL_NAME = os.getenv('MODEL_NAME', 'bert-base-uncased')
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

# --- Configuration for 3D placement ---
MIN_DEPTH_FACTOR = 0.4 

# --- Redis Configuration ---
REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', '')
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

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

# --- Redis Client ---
redis_client = None

# --- Load Model ---
tokenizer = None
model = None

class WordsRequest(BaseModel):
    words: List[str]

@app.on_event("startup")
async def startup_event():
    global tokenizer, model, redis_client
    
    # Initialize Redis client
    logger.info(f"Connecting to Redis at {REDIS_URL}...")
    try:
        if REDIS_PASSWORD:
            redis_client = redis.from_url(REDIS_URL, password=REDIS_PASSWORD, db=REDIS_DB)
        else:
            redis_client = redis.from_url(REDIS_URL, db=REDIS_DB)
        redis_client.ping()
        logger.info("Redis connection established successfully.")
    except Exception as e:
        logger.error(f"Error connecting to Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to connect to Redis: {e}")
    
    logger.info(f"Loading model: {MODEL_NAME}...")
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        model = BertModel.from_pretrained(MODEL_NAME)
        model.eval()
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Error loading model: {e}")

# --- Redis Helper Functions ---
def get_all_words_and_embeddings():
    """Retrieve all words and their embeddings from Redis"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    
    words = []
    embeddings = []
    
    all_words = redis_client.smembers('all_words')
    
    if not all_words:
        return [], []
    
    for word_bytes in all_words:
        word = word_bytes.decode('utf-8')
        embedding_key = f"embedding:{word}"
        embedding_bytes = redis_client.get(embedding_key)
        
        if embedding_bytes:
            try:
                embedding = pickle.loads(embedding_bytes)
                words.append(word)
                embeddings.append(embedding)
            except Exception as e:
                logger.error(f"Error unpickling embedding for word '{word}': {e}")
    
    logger.info(f"Retrieved {len(words)} words and embeddings from Redis")
    return words, embeddings

def add_words_to_redis(valid_words, word_vectors):
    """Add words and their embeddings to Redis"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    
    for word, vector in zip(valid_words, word_vectors):
        try:
            embedding_key = f"embedding:{word}"
            embedding_bytes = pickle.dumps(vector)
            redis_client.set(embedding_key, embedding_bytes)
            
            redis_client.sadd('all_words', word)
            logger.info(f"Added word '{word}' to Redis")
        except Exception as e:
            logger.error(f"Error adding word '{word}' to Redis: {e}")
    
    return True

def reset_redis_words():
    """Remove all words and embeddings from Redis"""
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis client not initialized")
    
    try:
        all_words = redis_client.smembers('all_words')
        
        pipe = redis_client.pipeline()
        
        for word_bytes in all_words:
            word = word_bytes.decode('utf-8')
            embedding_key = f"embedding:{word}"
            pipe.delete(embedding_key)
        
        pipe.delete('all_words')
        
        pipe.execute()
        
        logger.info("Reset all words and embeddings in Redis")
        return True
    except Exception as e:
        logger.error(f"Error resetting words in Redis: {e}")
        return False

def get_embeddings(words: list[str]):
    """Generate BERT embeddings for a list of words"""
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
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
    
    return valid_words, word_vectors

# --- Helper Function ---
def process_words(words: list[str], get_new_embeddings=True):
    """Process words to get visualization data"""
    if not tokenizer or not model:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    all_words, all_embeddings = get_all_words_and_embeddings()
    
    if get_new_embeddings and words:
        new_words = [word for word in words if word not in all_words]
        
        if new_words:
            logger.info(f"Processing new words: {new_words}")
            valid_new_words, new_vectors = get_embeddings(new_words)
            
            add_words_to_redis(valid_new_words, new_vectors)
            
            all_words, all_embeddings = get_all_words_and_embeddings()
    
    if not all_words:
        return {"words": [], "positions": [], "links": [], "clusters": []}
    
    word_vectors = np.array(all_embeddings)

    if len(all_words) == 1:
        return {"words": all_words, "positions": [[0, 0, 0]], "links": [], "clusters": [0]}

    # 2. Dimensionality Reduction (t-SNE)
    current_perplexity = min(PERPLEXITY, len(all_words) - 1)
    if current_perplexity <= 0:
        logger.warning("Not enough samples for t-SNE.")
        return {"words": all_words, "positions": [], "links": [], "clusters": []}

    tsne = TSNE(n_components=DIMENSIONS, random_state=42, perplexity=current_perplexity, max_iter=300, init='pca')
    try:
        word_vectors_3d = tsne.fit_transform(word_vectors)
    except ValueError as e:
        logger.error(f"t-SNE failed: {e}. Word vectors shape: {word_vectors.shape}")
        return {"words": all_words, "positions": [], "links": [], "clusters": []}

    # 3. Map to sphere or inside globe
    norms = np.linalg.norm(word_vectors_3d, axis=1, keepdims=True)
    norms[norms == 0] = 1e-8 
    
    word_vectors_sphere = []
    for i, vec in enumerate(word_vectors_3d):
        norm = norms[i][0]
        
        normalized_vec = vec / norm
        
        depth_factor = random.uniform(MIN_DEPTH_FACTOR, 1.0)
        
        scaled_radius = GLOBE_RADIUS * depth_factor
        
        position = normalized_vec * scaled_radius
        word_vectors_sphere.append(position)
        
    word_vectors_sphere = np.array(word_vectors_sphere)

    # 4. Clustering with AgglomerativeClustering
    if len(all_words) > 1:
        n_clusters = min(MAX_CLUSTERS, max(MIN_CLUSTERS, len(all_words) // 20))
        if len(all_words) < 10:
            n_clusters = max(1, len(all_words) // 2)
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            linkage='ward',
            metric='euclidean'
        )
        try:
            cluster_labels = clusterer.fit_predict(word_vectors)
        except Exception as e:
            logger.error(f"AgglomerativeClustering failed: {e}. Using single cluster.")
            cluster_labels = np.zeros(len(all_words), dtype=int)
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
    if len(all_words) > 1:
        similarity_matrix = cosine_similarity(word_vectors)
        centroid_similarity = cosine_similarity(centroids) if len(unique_clusters) > 1 else np.ones((1, 1))

        for i in range(len(all_words)):
            for j in range(i + 1, len(all_words)):
                similarity = similarity_matrix[i, j]
                if similarity < MIN_SIMILARITY:
                    logger.info(f"No link between '{all_words[i]}' and '{all_words[j]}' (similarity {similarity:.2f} < {MIN_SIMILARITY})")
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
                        logger.info(f"Created link between '{all_words[i]}' and '{all_words[j]}' (same cluster {cluster_i}, similarity {similarity:.2f})")
                    else:
                        logger.info(f"No link between '{all_words[i]}' and '{all_words[j]}' (same cluster {cluster_i}, similarity {similarity:.2f} < {WITHIN_CLUSTER_SIMILARITY})")
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
                        logger.info(f"Created link between '{all_words[i]}' and '{all_words[j]}' (cross-cluster {cluster_i} vs {cluster_j}, similarity {similarity:.2f}, cluster similarity {cluster_sim:.2f})")
                    else:
                        logger.info(f"No link between '{all_words[i]}' and '{all_words[j]}' (cross-cluster {cluster_i} vs {cluster_j}, low cluster similarity {cluster_sim:.2f} < {CLUSTER_DISTANCE_THRESHOLD})")
                else:
                    logger.info(f"No link between '{all_words[i]}' and '{all_words[j]}' (cross-cluster {cluster_i} vs {cluster_j}, similarity {similarity:.2f} not in {CROSS_CLUSTER_SIMILARITY_MIN}–{CROSS_CLUSTER_SIMILARITY_MAX})")

    logger.info(f"Processing complete. Found {len(links)} links across {len(unique_clusters)} clusters.")
    return {
        "words": all_words,
        "positions": word_vectors_sphere.tolist(),
        "links": links,
        "clusters": cluster_labels.tolist()
    }

# --- API Endpoints ---
@app.get("/api/word-data")
async def get_word_data():
    """Get visualization data for all stored words"""
    try:
        data = process_words([], get_new_embeddings=False)
        return data
    except Exception as e:
        logger.error(f"Error processing words in endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during word processing: {e}")

@app.post("/api/add-words")
async def add_words(request: WordsRequest):
    """Add new words and get updated visualization"""
    if not request.words:
        return {"words": [], "positions": [], "links": [], "clusters": []}
    
    words_list = [word.strip() for word in request.words if word.strip()]
    
    try:
        data = process_words(words_list)
        return data
    except Exception as e:
        logger.error(f"Error processing words in endpoint: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error during word processing: {e}")

@app.post("/api/reset")
async def reset_words():
    """Reset all stored words"""
    reset_redis_words()
    return {"message": "Word list reset successfully"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
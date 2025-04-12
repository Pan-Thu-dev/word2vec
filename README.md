# Word2Vec Visualization

This project visualizes word embeddings from a BERT model in a 3D interactive globe. Words that are semantically related appear closer together on the globe, and highly similar words are connected by lines.

## Project Structure

- `backend/`: Python FastAPI backend that provides word embeddings using the BERT model
- `frontend/`: Next.js frontend that visualizes the word embeddings in a 3D globe using Three.js

## Prerequisites

- **Python 3.8+**: Install from [python.org](https://python.org/)
- **Node.js and npm**: Install from [nodejs.org](https://nodejs.org/)
- **Git**: (Optional but recommended) for version control

## Setup

### Backend

1. Navigate to the backend directory:
   ```bash
   cd backend
   ```

2. Create and activate a virtual environment:
   ```bash
   # For Windows
   python -m venv venv
   venv\Scripts\activate

   # For macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Start the backend server:
   ```bash
   uvicorn main:app --reload --port 8000
   ```

### Frontend

1. Open a new terminal and navigate to the frontend directory:
   ```bash
   cd frontend
   ```

2. Install dependencies:
   ```bash
   npm install
   ```

3. Start the development server:
   ```bash
   npm run dev
   ```

4. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Usage

1. Enter comma-separated words in the input field.
2. Click the "Visualize Words" button.
3. The words will be displayed on a 3D globe, with related words positioned closer together.
4. Use your mouse to rotate, zoom, and pan the globe to explore the word relationships.

## How It Works

1. The frontend sends a list of words to the backend API.
2. The backend uses a BERT model to generate embeddings for each word.
3. These high-dimensional embeddings are reduced to 3D using t-SNE.
4. The 3D coordinates are projected onto a sphere (the globe).
5. Words with high similarity (based on cosine similarity) are connected by lines.
6. The frontend renders this data as an interactive 3D visualization. 
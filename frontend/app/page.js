'use client'; // Required for useState, useEffect

import React, { useState, useEffect, Suspense } from 'react';
import dynamic from 'next/dynamic';
import { Button, Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle, Input } from '@/components';

// Dynamically import the GlobeVisualization component to avoid SSR issues with Three.js
const GlobeVisualization = dynamic(
  () => import('@/components').then(mod => ({ default: mod.GlobeVisualization })),
  { ssr: false, loading: () => <div className="flex items-center justify-center h-[700px]"><p className="text-lg text-muted-foreground">Loading 3D Scene...</p></div> }
);

// Use environment variable for API URL
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export default function Home() {
  const [wordData, setWordData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [wordsInput, setWordsInput] = useState(""); 
  const [addedWords, setAddedWords] = useState([]); 

  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  // Fetch initial data on page load
  const fetchInitialData = async () => {
    setIsLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/api/word-data`);

      if (!response.ok) {
        let errorDetail = `HTTP error! Status: ${response.status}`;
        try {
          const errorBody = await response.json();
          errorDetail = errorBody.detail || errorDetail;
        } catch (jsonError) {
        }
        throw new Error(errorDetail);
      }

      const data = await response.json();
      console.log("Fetched initial data:", data);
      setWordData(data);
      setAddedWords(data.words || []);

    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message || "Failed to fetch word data.");
      setWordData(null);
    } finally {
      setIsLoading(false);
    }
  };

  // Add new words
  const addWords = async (newWords) => {
    if (!newWords || newWords.trim() === '') return;
    
    setIsLoading(true);
    setError(null);
    
    try {
      const wordsToAdd = newWords.split(',')
        .map(word => word.trim())
        .filter(word => word !== '');
      
      const response = await fetch(`${API_BASE_URL}/api/add-words`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ words: wordsToAdd }),
      });

      if (!response.ok) {
        let errorDetail = `HTTP error! Status: ${response.status}`;
        try {
          const errorBody = await response.json();
          errorDetail = errorBody.detail || errorDetail;
        } catch (jsonError) {
        }
        throw new Error(errorDetail);
      }

      const data = await response.json();
      console.log("Added words, new data:", data);
      setWordData(data);
      setAddedWords(data.words || []);
      setWordsInput(''); 

    } catch (err) {
      console.error("Error adding words:", err);
      setError(err.message || "Failed to add words.");
    } finally {
      setIsLoading(false);
    }
  };

  // Reset all words
  const resetWords = async () => {
    setIsLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/reset`, {
        method: 'POST',
      });

      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }

      await fetchInitialData(); // Refresh the view
      
    } catch (err) {
      console.error("Error resetting words:", err);
      setError(err.message || "Failed to reset words.");
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch data on initial load
  useEffect(() => {
    fetchInitialData();
  }, []);

  const handleInputChange = (event) => {
    setWordsInput(event.target.value);
  };

  const handleAddWords = () => {
    addWords(wordsInput);
  };

  return (
    <div className="container py-10 space-y-8">
      <div className="flex flex-col space-y-2">
        <h1 className="text-4xl font-bold tracking-tight">Interactive Word Embedding Globe</h1>
        <p className="text-muted-foreground">
          Visualize word relationships in 3D space using word2vec embeddings
        </p>
      </div>
      
      <Card>
        <CardHeader>
          <CardTitle>Word Selection</CardTitle>
          <CardDescription>
            Enter comma-separated words to add to the visualization
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col space-y-4">
            <div className="flex flex-col space-y-2 sm:flex-row sm:space-y-0 sm:space-x-2">
              <Input
                id="wordsInput"
                value={wordsInput}
                onChange={handleInputChange}
                className="sm:min-w-[300px]"
                placeholder="Enter words separated by commas"
              />
              <Button onClick={handleAddWords} disabled={isLoading || !wordsInput.trim()}>
                {isLoading ? 'Processing...' : 'Add Words'}
              </Button>
              <Button onClick={resetWords} disabled={isLoading} variant="outline">
                Reset All
              </Button>
            </div>
            
            {error && (
              <p className="text-sm text-destructive">{error}</p>
            )}
            
            {addedWords.length > 0 && (
              <div className="mt-2">
                <h4 className="text-sm font-medium mb-1">Current Words:</h4>
                <div className="flex flex-wrap gap-1">
                  {addedWords.map((word, index) => (
                    <span key={index} className="inline-flex items-center px-2 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary">
                      {word}
                    </span>
                  ))}
                </div>
              </div>
            )}
          </div>
        </CardContent>
      </Card>

      <Card className="overflow-hidden">
        <CardHeader className="pb-0">
          <CardTitle>3D Visualization</CardTitle>
          <CardDescription>
            Explore word relationships by interacting with the 3D globe
          </CardDescription>
        </CardHeader>
        <CardContent className="p-0">
          <div className="h-[800px] w-full relative flex items-center justify-center">
            {(isLoading || wordData) && !error ? (
              <Suspense fallback={<div className="flex items-center justify-center h-full"><p className="text-lg text-muted-foreground">Loading...</p></div>}>
                <GlobeVisualization wordData={wordData} />
              </Suspense>
            ) : !isLoading && !error && !wordData ? (
              <div className="flex items-center justify-center h-full">
                <p className="text-muted-foreground">No data loaded. Add some words to begin.</p>
              </div>
            ) : null}
          </div>
        </CardContent>
        <CardFooter className="border-t px-6 py-4">
          <p className="text-xs text-muted-foreground">
            Tip: Drag to rotate, scroll to zoom, and double-click words to explore their relationships
          </p>
        </CardFooter>
      </Card>
    </div>
  );
} 
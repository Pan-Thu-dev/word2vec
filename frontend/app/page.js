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
  const [wordsInput, setWordsInput] = useState("dog,cat,puppy,kitten,lion,tiger,wolf,fox,bear,cheetah"); // Default input

  // Set dark mode permanently
  useEffect(() => {
    document.documentElement.classList.add('dark');
  }, []);

  const fetchData = async (wordsQuery) => {
    setIsLoading(true);
    setError(null);
    try {
      // Encode the query parameter
      const encodedQuery = encodeURIComponent(wordsQuery);
      const response = await fetch(`${API_BASE_URL}/api/word-data?words_query=${encodedQuery}`);

      if (!response.ok) {
        // Try to get error message from backend response body
        let errorDetail = `HTTP error! Status: ${response.status}`;
        try {
          const errorBody = await response.json();
          errorDetail = errorBody.detail || errorDetail;
        } catch (jsonError) {
          // Ignore if response is not JSON
        }
        throw new Error(errorDetail);
      }

      const data = await response.json();
      console.log("Fetched data:", data); // Log fetched data
      setWordData(data);

    } catch (err) {
      console.error("Fetch error:", err);
      setError(err.message || "Failed to fetch word data.");
      setWordData(null); // Clear data on error
    } finally {
      setIsLoading(false);
    }
  };

  // Fetch data on initial load with default words
  useEffect(() => {
    fetchData(wordsInput);
  }, []); // Fetch only on initial mount with default words

  const handleInputChange = (event) => {
    setWordsInput(event.target.value);
  };

  const handleFetchClick = () => {
    fetchData(wordsInput);
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
            Enter comma-separated words to visualize their relationships in vector space
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col space-y-2 sm:flex-row sm:space-y-0 sm:space-x-2">
            <Input
              id="wordsInput"
              value={wordsInput}
              onChange={handleInputChange}
              className="sm:min-w-[300px]"
              placeholder="Enter words separated by commas"
            />
            <Button onClick={handleFetchClick} disabled={isLoading}>
              {isLoading ? 'Processing...' : 'Visualize Words'}
            </Button>
          </div>
          {error && (
            <p className="mt-2 text-sm text-destructive">{error}</p>
          )}
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
                <p className="text-muted-foreground">No data loaded. Check input or backend connection.</p>
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
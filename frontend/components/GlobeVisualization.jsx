'use client'; // Needed for components using hooks like useState, useEffect

import React, { Suspense, useRef, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Sphere, Text, Line } from '@react-three/drei';

// Rotating globe animation component
function RotatingGlobe() {
  const groupRef = useRef();
  
  useFrame(({ clock }) => {
    if (groupRef.current) {
      // Slow automatic rotation
      groupRef.current.rotation.y = clock.getElapsedTime() * 0.05;
    }
  });
  
  return null; // This is just for the rotation logic
}

// Placeholder component for the actual visualization
function SceneContent({ words, positions, links }) {
    const globeRadius = 10; // Should match backend or be dynamic
    const groupRef = useRef();

    return (
        <>
            {/* Optional: Add lighting */}
            <ambientLight intensity={0.8} />  
            <pointLight position={[20, 20, 20]} intensity={1} />
            <pointLight position={[-20, -20, -20]} intensity={0.5} />

            {/* Group for all elements that will rotate together */}
            <group ref={groupRef}>
                {/* The Globe Wireframe */}
                <Sphere args={[globeRadius, 48, 48]} position={[0, 0, 0]}>
                    <meshBasicMaterial wireframe color="lightblue" opacity={0.8} transparent />
                </Sphere>

                {/* Render Word Points and Text */}
                {positions.map((pos, index) => (
                    <group key={index} position={pos}>
                        {/* Small sphere marker for the word */}
                        <Sphere args={[0.2, 16, 16]}>
                            <meshStandardMaterial color="red" />
                        </Sphere>
                        {/* Text Label - adjust position/size as needed */}
                        <Text
                            position={[0, 0.5, 0]} // Offset text slightly above the point
                            fontSize={0.6}
                            color="white"
                            anchorX="center"
                            anchorY="middle"
                            renderOrder={1} // Ensure text renders on top
                            depthTest={false} // Make sure text is always visible
                        >
                            {words[index] || ''} {/* Ensure word exists */}
                        </Text>
                    </group>
                ))}

                {/* Render Links */}
                {links.map(([startIndex, endIndex], linkIndex) => {
                    // Ensure positions exist before trying to render line
                    if (positions[startIndex] && positions[endIndex]) {
                        return (
                            <Line
                                key={`link-${linkIndex}`}
                                points={[positions[startIndex], positions[endIndex]]}
                                color="white"
                                lineWidth={1}
                                opacity={0.6}
                                transparent
                            />
                        );
                    }
                    return null; // Don't render line if points are missing
                })}
            </group>
            
            <RotatingGlobe />
        </>
    );
}

// Main component that sets up Canvas and fetches data
export default function GlobeVisualization({ wordData }) {
    if (!wordData || !wordData.positions) {
        return <div className="flex items-center justify-center w-full h-full">
            <p className="text-lg text-muted-foreground">Loading data or data format error...</p>
        </div>;
    }

    const { words = [], positions = [], links = [] } = wordData;

    // Basic check if data seems valid
    const hasData = words.length > 0 && positions.length > 0 && words.length === positions.length;

    return (
        <div className="w-full h-full">
            <Canvas camera={{ position: [0, 0, 22], fov: 45 }}>
                {/* Suspense allows fallback while async components load */}
                <Suspense fallback={null}>
                    {hasData ? (
                        <SceneContent words={words} positions={positions} links={links} />
                    ) : (
                        <Text position={[0, 0, 0]} fontSize={1} color="orange">No valid word data to display.</Text>
                    )}
                </Suspense>
                <OrbitControls enableZoom={true} enablePan={true} minDistance={15} maxDistance={30} />
            </Canvas>
        </div>
    );
} 
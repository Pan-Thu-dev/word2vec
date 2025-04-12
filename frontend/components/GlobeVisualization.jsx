'use client'; // Needed for components using hooks like useState, useEffect

import React, { Suspense, useRef, useEffect, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Sphere, Text, Line, Billboard, Sparkles, MeshDistortMaterial, MeshWobbleMaterial } from '@react-three/drei';
import * as THREE from 'three';

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

// Camera controller for smooth transitions
function CameraController({ target, resetZoom }) {
  const { camera, controls } = useThree();
  
  // Store initial camera position for reset
  const initialPosition = useRef(new THREE.Vector3(0, 0, 22));
  const initialTarget = useRef(new THREE.Vector3(0, 0, 0));
  
  useEffect(() => {
    if (resetZoom && controls) {
      // Reset to initial position
      controls.target.copy(initialTarget.current);
      camera.position.copy(initialPosition.current);
      controls.update();
    } else if (target && target.length === 3 && controls) {
      // Create Vector3 from target position
      const targetVector = new THREE.Vector3(target[0], target[1], target[2]);
      
      // Set new target for OrbitControls
      controls.target.copy(targetVector);
      
      // Position camera closer to the target
      const zoomDistance = 3;
      const direction = targetVector.clone().normalize();
      const cameraPosition = targetVector.clone().add(
        direction.multiplyScalar(zoomDistance)
      );
      
      // Set camera position
      camera.position.copy(cameraPosition);
      
      // Update controls
      controls.update();
    }
  }, [target, resetZoom, camera, controls]);
  
  return null;
}

// 3D Node component with glow effect
function WordNode({ position, word, onClick }) {
  // Reference for animation
  const sphereRef = useRef();
  
  // Subtle floating animation for the node
  useFrame(({ clock }) => {
    if (sphereRef.current) {
      // Add a subtle bobbing motion
      const time = clock.getElapsedTime();
      sphereRef.current.position.y = Math.sin(time * 0.8) * 0.1;
    }
  });

  return (
    <group position={position} onClick={(e) => {
      // Stop propagation to prevent globe click
      e.stopPropagation();
      onClick(position, word);
    }}>
      {/* Enhanced 3D sphere */}
      <group ref={sphereRef}>
        {/* Core sphere with glowing material - make it clickable */}
        <Sphere args={[0.15, 32, 32]}>
          <MeshDistortMaterial
            color="#ff3333"
            distort={0.2} // Amount of distortion
            speed={1.5} // Speed of distortion
            roughness={0.2}
            metalness={0.8}
            emissive="#ff0000"
            emissiveIntensity={0.6}
          />
        </Sphere>
        
        {/* Outer glow effect */}
        <Sphere args={[0.25, 16, 16]}>
          <meshBasicMaterial color="#ff0000" transparent opacity={0.15} />
        </Sphere>
      </group>
      
      {/* Particle effects around the node */}
      <Sparkles 
        count={25}
        scale={[1.5, 1.5, 1.5]}
        size={0.5}
        speed={0.3}
        color="#ff6666"
        opacity={0.7}
      />
      
      {/* Billboard ensures text always faces camera */}
      <Billboard follow={true} lockX={false} lockY={false} lockZ={false}>
        <Text
          position={[0, 0.9, 0]} // Offset text slightly above the point
          fontSize={0.7}
          color="white"
          anchorX="center"
          anchorY="middle"
          renderOrder={2} // Ensure text renders on top
          depthTest={false} // Make sure text is always visible
          outlineWidth={0.05}
          outlineColor="#000000"
        >
          {word}
        </Text>
      </Billboard>
    </group>
  );
}

// Placeholder component for the actual visualization
function SceneContent({ words, positions, links }) {
    const globeRadius = 10; // Should match backend or be dynamic
    const groupRef = useRef();
    const [targetPosition, setTargetPosition] = useState(null);
    const [resetZoom, setResetZoom] = useState(false);

    // Handle node click
    const handleNodeClick = (position, word) => {
      console.log(`Clicked on word: ${word}`);
      setTargetPosition(position);
      setResetZoom(false);
    };

    // Handle globe background click to reset zoom
    const handleGlobeClick = (e) => {
      // Only reset if we're not clicking on a node
      if (e.object.type === 'Mesh' && e.object.geometry.type === 'SphereGeometry') {
        console.log('Reset zoom');
        setResetZoom(true);
        setTargetPosition(null);
      }
    };

    // Function to generate distinct colors for relationships based on word pairs
    const getRelationshipColor = (startIndex, endIndex) => {
      // Create some distinct colors for relationships
      const colors = [
        "#4cc9f0", // blue
        "#f72585", // pink
        "#7209b7", // purple
        "#4361ee", // indigo
        "#3a0ca3", // deep purple
        "#4895ef", // light blue
        "#560bad", // violet
        "#f3722c", // orange
        "#90be6d", // green
        "#43aa8b", // teal
      ];
      
      // Use a simple hash to get a consistent color for each word pair
      const colorIndex = (startIndex * 13 + endIndex * 17) % colors.length;
      return colors[colorIndex];
    };

    return (
        <>
            {/* Camera controller for smooth transitions */}
            <CameraController target={targetPosition} resetZoom={resetZoom} />
            
            {/* Enhanced lighting for better 3D appearance */}
            <ambientLight intensity={0.5} />  
            <pointLight position={[20, 20, 20]} intensity={1.5} />
            <pointLight position={[-20, -20, -20]} intensity={0.8} />
            <pointLight position={[0, 30, 0]} intensity={1} />
            <pointLight position={[0, -30, 0]} intensity={0.6} />

            {/* Globe Wireframe - static, doesn't rotate - clickable for reset */}
            <Sphere args={[globeRadius, 64, 64]} position={[0, 0, 0]} onClick={handleGlobeClick}>
                <meshBasicMaterial wireframe wireframeLinewidth={0.5} color="#30404d" opacity={0.3} transparent />
            </Sphere>

            {/* Group for all relationship lines */}
            <group ref={groupRef}>
                {/* Render Links with distinct colors */}
                {links.map(([startIndex, endIndex], linkIndex) => {
                    // Ensure positions exist before trying to render line
                    if (positions[startIndex] && positions[endIndex]) {
                        const lineColor = getRelationshipColor(startIndex, endIndex);
                        return (
                            <Line
                                key={`link-${linkIndex}`}
                                points={[positions[startIndex], positions[endIndex]]}
                                color={lineColor}
                                lineWidth={2.5} // Make lines thicker than the globe wireframe
                                opacity={0.8}
                                transparent
                                renderOrder={1} // Render above globe wireframe
                            />
                        );
                    }
                    return null; // Don't render line if points are missing
                })}
            </group>
            
            {/* Word nodes - separate from rotation group so they remain fixed */}
            {positions.map((pos, index) => (
                <WordNode 
                    key={`word-${index}`}
                    position={pos} 
                    word={words[index] || ''}
                    onClick={handleNodeClick}
                />
            ))}
            
            {/* Apply rotation only to the lines */}
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
                <OrbitControls 
                  enableZoom={true} 
                  enablePan={true} 
                  minDistance={2} 
                  maxDistance={30}
                  makeDefault
                />
            </Canvas>
        </div>
    );
} 
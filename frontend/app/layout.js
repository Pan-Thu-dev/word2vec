import React from "react";
import { Inter, Roboto_Mono } from "next/font/google";
import "./globals.css";

const inter = Inter({
  variable: "--font-sans",
  subsets: ["latin"],
});

const robotoMono = Roboto_Mono({
  variable: "--font-mono",
  subsets: ["latin"],
});

export const metadata = {
  title: "Word Embedding Visualization",
  description: "Interactive 3D visualization of word embeddings",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en" className="dark" suppressHydrationWarning>
      <body className={`${inter.variable} ${robotoMono.variable} min-h-screen bg-background font-sans antialiased`}>
        <div className="relative flex min-h-screen flex-col">
          <header className="sticky top-0 z-50 w-full border-b bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60">
            <div className="container flex h-14 items-center">
              <div className="mr-4 flex font-bold text-xl">
                <a href="/" className="mr-2">Word2Vec Explorer</a>
              </div>
              <nav className="flex items-center space-x-4 text-sm font-medium">
                <a href="/" className="transition-colors hover:text-foreground/80">Home</a>
                <a href="https://github.com/your-repo/word2vec" className="transition-colors hover:text-foreground/80" target="_blank" rel="noopener noreferrer">GitHub</a>
              </nav>
            </div>
          </header>
          <main className="flex-1">{children}</main>
          <footer className="border-t py-6 md:py-0">
            <div className="container flex flex-col items-center justify-between gap-4 md:h-14 md:flex-row">
              <p className="text-center text-sm leading-loose text-muted-foreground md:text-left">
                Built with Next.js, Three.js and word2vec. © {new Date().getFullYear()}.
              </p>
            </div>
          </footer>
        </div>
      </body>
    </html>
  );
} 
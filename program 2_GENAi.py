PROGRAM -2 
 
Use dimensionality reduction (e.g., PCA or t-SNE) to visualize word embeddings 
for Q 1. Select 10 words from a specific domain (e.g., sports, technology) and 
visualize their embeddings. Analyze clusters and relationships. Generate 
contextually rich outputs using embeddings. Write a program to generate 5 
semantically similar words for a given input. 
 
import gensim.downloader as api 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.decomposition import PCA 
 
# Load pre-trained Word2Vec model (Google News) 
print("Loading model... (This may take a while)") 
model = api.load("word2vec-google-news-300") 
print("Model loaded!") 
 
# Select 10 words from the Technology domain 
tech_words = ["computer", "algorithm", "software", "hardware", "AI",  
              "cloud", "database", "network", "cybersecurity", "encryption"] 
 
# Get their word vectors 
word_vectors = np.array([model[word] for word in tech_words]) 
 
# Perform PCA to reduce to 2D 
pca = PCA(n_components=2) 
reduced_vectors = pca.fit_transform(word_vectors) 
 
# Plot the words in 2D 
plt.figure(figsize=(8,6)) 
for word, (x, y) in zip(tech_words, reduced_vectors): 
    plt.scatter(x, y) 
    plt.text(x+0.02, y+0.02, word, fontsize=12) 
 
plt.title("2D Visualization of Technology Word Embeddings") 
plt.xlabel("PCA Component 1") 
plt.ylabel("PCA Component 2") 
plt.grid() 
plt.show() 
 
# Function to find 5 similar words 
def find_similar_words(word): 
    try: 
        similar_words = model.most_similar(word, topn=5) 
        print(f"\n5 words similar to '{word}':") 
        for w, score in similar_words: 
            print(f"{w}: {score:.4f}") 
    except KeyError: 
        print(f"'{word}' not found in the vocabulary.") 
 
# Test with an input word 
find_similar_words("AI")
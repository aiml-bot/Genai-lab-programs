PROGRAM -4 
 
Use word embeddings to improve prompts for Generative AI model. Retrieve 
similar words using word embeddings. Use the similar words to enrich a 
GenAI prompt. Use the AI model to generate responses for the original and 
enriched prompts. Compare the outputs in terms of detail and relevance. 
 
Follow the steps to run the program if error occurs 
 
step1 : - Update the opt tree  
python -m pip install --upgrade "optree>=0.13.0" 
 
 step 2: install tf keras library 
pip install tf-keras 
 
step3:- Fix TensorFlow one DNN Warnings run the program in VSCODE termninal 
 
set TF_ENABLE_ONEDNN_OPTS=0  # Windows Command Prompt 
 
Step 4 :- check the required libraries 
pip install --upgrade transformers gensim torch tensorflow optree 
 
Now run the program 
 
import gensim.downloader as api 
from transformers import pipeline 
 
# Load embedding model 
embedding_model = api.load("glove-wiki-gigaword-100") 
 
original_prompt = "Describe the beautiful landscapes during sunset." 
 
def enrich_prompt(prompt, embedding_model, n=5): 
    words = prompt.split() 
    enriched_prompt = [] 
     
    for word in words: 
        word_lower = word.lower() 
         
        if word_lower in embedding_model: 
            similar_words = embedding_model.most_similar(word_lower, topn=n) 
            similar_word_list = [w[0] for w in similar_words] 
            enriched_prompt.append(" ".join(similar_word_list))  # Join similar words as a phrase 
        else: 
            enriched_prompt.append(word)  # Keep the word as is if not found 
     
    return " ".join(enriched_prompt) 
 
enriched_prompt = enrich_prompt(original_prompt, embedding_model) 
 
# Load text generation model 
generator = pipeline("text-generation", model="gpt2") 
 
# Generate responses 
original_response = generator(original_prompt, max_length=50, num_return_sequences=1) 
enriched_response = generator(enriched_prompt, max_length=50, 
num_return_sequences=1) 
 
# Print results 
print("Original prompt response") 
print(original_response[0]['generated_text']) 
 
print("\nEnriched prompt response") 
print(enriched_response[0]['generated_text'])
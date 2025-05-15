PROGRAM -7 
Summarize long texts using a pre-trained summarization model using Hugging 
face model. Load the summarization pipeline. Take a passage as input and 
obtain the summarized text.  
 
from transformers import pipeline 
 
# Load the summarization model 
summarizer = pipeline("summarization", model="facebook/bart-large-cnn") 
 
# Take user input for the text passage 
text = input("Enter the text you want to summarize:\n") 
 
# Summarize the text 
summary = summarizer(text, max_length=100, min_length=30, do_sample=False) 
 
# Print the summarized text 
print("\nSummarized Text:") 
print(summary[0]['summary_text'])
PROGRAM -7 
Summarize long texts using a pre-trained summarization model using Hugging 
face model. Load the summarization pipeline. Take a passage as input and 
obtain the summarized text.  
 
# First, install the required dependencies
!pip install transformers
!pip install torch  # Installing PyTorch

# After installation, import the necessary libraries
from transformers import pipeline

# Define the summarization function
def summarize_text(text, max_length=150, min_length=None):
    # Initialize the summarizer
    summarizer = pipeline("summarization")
    
    if not min_length:
        min_length = max(30, max_length // 3)

    # Default summarization
    summary_1 = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)

    # High randomness (creative output)
    summary_2 = summarizer(text, max_length=max_length, min_length=min_length, do_sample=True, temperature=2.9)

    # Conservative approach (structured)
    summary_3 = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False, num_beams=5)

    # Diverse sampling (top-k and top-p)
    summary_4 = summarizer(text, max_length=max_length, min_length=min_length, do_sample=True, top_k=50, top_p=0.95)

    print("\nOriginal Text:")
    print(text)

    print("\nSummarized Texts:")
    print("Default:", summary_1[0]['summary_text'])
    print("High Randomness:", summary_2[0]['summary_text'])
    print("Conservative:", summary_3[0]['summary_text'])
    print("Diverse Sampling:", summary_4[0]['summary_text'])

# Example long text passage
long_text = """
Artificial Intelligence (AI) is a rapidly evolving field of computer science focused on creating 
intelligent machines capable of mimicking human cognitive functions such as learning, problem-solving, 
and decision-making. In recent years, AI has significantly impacted various industries, including healthcare, 
finance, education, and entertainment. AI-powered applications, such as chatbots, self-driving cars, and 
recommendation systems, have transformed the way we interact with technology. Machine learning and deep learning, 
subsets of AI, enable systems to learn from data and improve over time without explicit programming. However, AI 
also poses ethical challenges, such as bias in decision-making and concerns over job displacement. As AI 
technology continues to advance, it is crucial to balance innovation with ethical considerations to ensure its 
responsible development and deployment.
"""

# Summarize the passage
summarize_text(long_text)

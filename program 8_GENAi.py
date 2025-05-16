PROGRAM -8  
Install langchain, cohere (for key), langchain-community. Get the api key( By 
logging into Cohere and obtaining the cohere key). Load a text document from 
your google drive . Create a prompt template to display the output in a 
particular manner 
 
# First, install the required packages
!pip install langchain-community cohere

import cohere
import getpass

from langchain.prompts import PromptTemplate
from langchain_community.llms import Cohere  # This import requires langchain-community package

file_path = "Teaching.txt"  # Corrected relative pathname

try:
    with open(file_path, "r", encoding="utf-8") as file:
        text_content = file.read()
    print("File loaded successfully!")
except Exception as e:
    print("Error loading file:", str(e))


COHERE_API_KEY = getpass.getpass("Enter your Cohere API Key: ")


cohere_llm = Cohere(cohere_api_key=COHERE_API_KEY, model="command")


template = """
You are an AI assistant helping to summarize and analyze a text document.

Here is the document content:
{text}

Summary:
Provide a concise summary of the document.

Key Takeaways:
List 3 important points from the text.

Sentiment Analysis:
Determine if the sentiment of the document is Positive, Negative, or Neutral.
"""

prompt_template = PromptTemplate(input_variables=["text"], template=template)


formatted_prompt = prompt_template.format(text=text_content)
response = cohere_llm.predict(formatted_prompt)


print("\n*Formatted Output*")
print(response)

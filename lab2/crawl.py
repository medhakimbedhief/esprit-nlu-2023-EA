import requests
from bs4 import BeautifulSoup
import os

# Function to crawl and save text from a website
def crawl_website(url):
    # Send a GET request to the URL
    response = requests.get(url)

    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Extract all the text from the HTML content
    text = soup.get_text()

    # Save the text to a file with the same name as the domain name
    domain_name = url.split('//')[1].split('/')[0]
    file_name = f"{domain_name}.txt"
    with open(file_name, 'w', encoding='utf-8') as f:
        f.write(text)

    print(f"Text saved from {url} to {file_name}")


# List of URLs to crawl and save text
urls = ['http://fr.tunisie.gov.tn/', 'http://www.finances.gov.tn/fr', 'http://www.pm.gov.tn/pm/content/index.php']

# Loop through each URL and crawl and save the text
for url in urls:
    crawl_website(url)

# Fine-tuning a QA model on the crawled text
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# Load the data from the crawled text files
data = []
for file in os.listdir():
    if file.endswith(".txt"):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
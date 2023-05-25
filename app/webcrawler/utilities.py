################################################################################
### Step 1
################################################################################
from dotenv import dotenv_values
import requests
import re
import urllib.request
from bs4 import BeautifulSoup
from collections import deque
from html.parser import HTMLParser
from urllib.parse import urlparse
from config import Config
import os
import pandas as pd
import tiktoken
import openai
import pinecone
import numpy as np
from openai.embeddings_utils import distances_from_embeddings, cosine_similarity
from uuid import uuid4
import gzip
from tqdm.auto import tqdm
import shutil

config = dotenv_values(".env")
openai.api_key = config["OPENAI_API_KEY"]
PINECONE_API_KEY = config["PINECONE_KEY"]
PINECONE_API_ENV = config["PINECONE_ENV"]


# Regex pattern to match a URL
HTTP_URL_PATTERN = r'^http[s]{0,1}://.+$'

# Helper to check namespace
def extract_domain_name(domain):
    # Remove the top-level domain (e.g., ".ca") using the split() method
    domain_parts = domain.split(".")
    domain_name = domain_parts[0]
    return domain_name

# check if namespace exists
def check_namespace(domain):
    namespace = extract_domain_name(domain)
    # Initialize connection to Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
     # Define index name
    index_name = 'quickquacktest'

    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    index_stats = index.describe_index_stats()
    # Check if namespace exists
    if namespace in index_stats['namespaces']:
        return True
    else:
        return False

# Create a class to parse the HTML and get the hyperlinks
class HyperlinkParser(HTMLParser):
    def __init__(self):
        super().__init__()
        # Create a list to store the hyperlinks
        self.hyperlinks = []

    # Override the HTMLParser's handle_starttag method to get the hyperlinks
    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        # If the tag is an anchor tag and it has an href attribute, add the href attribute to the list of hyperlinks
        if tag == "a" and "href" in attrs:
            self.hyperlinks.append(attrs["href"])

################################################################################
### Step 2
################################################################################

# Function to get the hyperlinks from a URL
def get_hyperlinks(url):
    
    # Try to open the URL and read the HTML
    try:
        # Open the URL and read the HTML
        with urllib.request.urlopen(url) as response:

            # If the response is not HTML, return an empty list
            if not response.info().get('Content-Type').startswith("text/html"):
                return []
            
            # Get the encoding from the response header
            encoding = response.headers.get_content_charset()
            if encoding is None:
                encoding = 'utf-8'

            if response.info().get('Content-Encoding') == 'gzip':
                with gzip.GzipFile(fileobj=io.BytesIO(response.read())) as decompressed_file:
                    html = decompressed_file.read().decode(encoding)
            else:
                html = response.read().decode(encoding)

    except Exception as e:
        print(e)
        return []

    # Create the HTML Parser and then Parse the HTML to get hyperlinks
    parser = HyperlinkParser()
    parser.feed(html)

    return parser.hyperlinks

################################################################################
### Step 3
################################################################################

# Function to get the hyperlinks from a URL that are within the same domain
def get_domain_hyperlinks(local_domain, url):
    clean_links = []
    for link in set(get_hyperlinks(url)):
        clean_link = None

        # If the link is a URL, check if it is within the same domain
        if re.search(HTTP_URL_PATTERN, link):
            # Parse the URL and check if the domain is the same
            url_obj = urlparse(link)
            if url_obj.netloc == local_domain:
                clean_link = link

        # If the link is not a URL, check if it is a relative link
        else:
            if link.startswith("/"):
                link = link[1:]
            elif (
                link.startswith("#")
                or link.startswith("mailto:")
                or link.startswith("tel:")
            ):
                continue
            clean_link = "https://" + local_domain + "/" + link

        if clean_link is not None:
            if clean_link.endswith("/"):
                clean_link = clean_link[:-1]
            clean_links.append(clean_link)

    # Return the list of hyperlinks that are within the same domain
    return list(set(clean_links))


################################################################################
### Step 4
################################################################################

def crawl(url):
    # Parse the URL and get the domain
        
    local_domain = urlparse(url).netloc

    # Create a queue to store the URLs to crawl
    queue = deque([url])

    # Create a set to store the URLs that have already been seen (no duplicates)
    seen = set([url])

    # Create a directory to store the text files
    # change this
    if not os.path.exists("app/webcrawler/text/"):
            os.mkdir("app/webcrawler/text/")

    if not os.path.exists("app/webcrawler/text/"+local_domain+"/"):
            os.mkdir("app/webcrawler/text/" + local_domain + "/")

    # Create a directory to store the csv files
    if not os.path.exists("app/webcrawler/processed"):
            os.mkdir("app/webcrawler/processed")

    # While the queue is not empty, continue crawling
    while queue:

        # Get the next URL from the queue
        url = queue.pop()
        print(url) # for debugging and to see the progress
        
        # Save text from the url to a <url>.txt file
        with open('app/webcrawler/text/'+local_domain+'/'+url[8:].replace("/", "_") + ".txt", "w", encoding="UTF-8") as f:
        
            # Get the text from the URL using BeautifulSoup
            try:
                soup = BeautifulSoup(requests.get(url).text, "html.parser")
                # Get the text but remove the tags
                text = soup.get_text()
                # If the crawler gets to a page that requires JavaScript, it will stop the crawl
                if ("You need to enable JavaScript to run this app." in text):
                    print("Unable to parse page " + url + " due to JavaScript being required")
                
                # Otherwise, write the text to the file in the text directory
                f.write(text)
            except Exception as e:
                print(f"Error occurred while parsing: {e}")
            
        # Get the hyperlinks from the URL and add them to the queue
        for link in get_domain_hyperlinks(local_domain, url):
            if link not in seen:
                queue.append(link)
                seen.add(link)

  #Remove this

################################################################################
### Step 5
################################################################################

def remove_newlines(serie):
    serie = serie.str.replace('\n', ' ')
    serie = serie.str.replace('\\n', ' ')
    serie = serie.str.replace('  ', ' ')
    serie = serie.str.replace('  ', ' ')
    return serie

################################################################################
### Step 6
################################################################################

# Create a list to store the text files
def prep_df(domain):
    directory_path = "app/webcrawler/text/" + "www." + domain
    
    if not os.path.exists(directory_path):
        print(f"Directory does not exist: {directory_path}")
        return
    
    texts=[]

    try:
        for file in os.listdir(directory_path):
            with open(directory_path + "/" + file, "r", encoding="UTF-8") as f:
                text = f.read()
                texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))
    except FileNotFoundError:
        print(f"Directory {directory_path} not found")
# Get all the text files in the text directory
    # for file in os.listdir("app/webcrawler/text/" + domain):

    #     # Open the file and read the text
    #     with open("app/webcrawler/text/" + domain + "/" + file, "r", encoding="UTF-8") as f:
    #         text = f.read()

    #         # Omit the first 11 lines and the last 4 lines, then replace -, _, and #update with spaces.
    #         texts.append((file[11:-4].replace('-',' ').replace('_', ' ').replace('#update',''), text))

    # Create a dataframe from the list of texts
    df = pd.DataFrame(texts, columns = ['fname', 'text'])

    # Set the text column to be the raw text with the newlines removed
    df['text'] = df.fname + ". " + remove_newlines(df.text)
    df.to_csv('app/webcrawler/processed/scraped.csv')


################################################################################
### Step 7
################################################################################

# Load the cl100k_base tokenizer which is designed to work with the ada-002 model
    tokenizer = tiktoken.get_encoding("cl100k_base")

    df = pd.read_csv('app/webcrawler/processed/scraped.csv', index_col=0)
    df.columns = ['title', 'text']

# Tokenize the text and save the number of tokens to a new column
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))

    

################################################################################
### Step 8
################################################################################


# Function to split the text into chunks of a maximum number of tokens
    def split_into_many(text, max_tokens = 500):

        # Split the text into sentences
        sentences = text.split('. ')

        # Get the number of tokens for each sentence
        n_tokens = [len(tokenizer.encode(" " + sentence)) for sentence in sentences]
        
        chunks = []
        tokens_so_far = 0
        chunk = []

        # Loop through the sentences and tokens joined together in a tuple
        for sentence, token in zip(sentences, n_tokens):

            # If the number of tokens so far plus the number of tokens in the current sentence is greater 
            # than the max number of tokens, then add the chunk to the list of chunks and reset
            # the chunk and tokens so far
            if tokens_so_far + token > max_tokens:
                chunks.append(". ".join(chunk) + ".")
                chunk = []
                tokens_so_far = 0

            # If the number of tokens in the current sentence is greater than the max number of 
            # tokens, go to the next sentence
            if token > max_tokens:
                continue

            # Otherwise, add the sentence to the chunk and add the number of tokens to the total
            chunk.append(sentence)
            tokens_so_far += token + 1
            
        # Add the last chunk to the list of chunks
        if chunk:
            chunks.append(". ".join(chunk) + ".")

        return chunks
    

    shortened = []

    # Loop through the dataframe
    for row in df.iterrows():

        # If the text is None, go to the next row
        if row[1]['text'] is None:
            continue

        # If the number of tokens is greater than the max number of tokens, split the text into chunks
        if row[1]['n_tokens'] > 500:
            shortened += split_into_many(row[1]['text'])
        
        # Otherwise, add the text to the list of shortened texts
        else:
            shortened.append( row[1]['text'] )

################################################################################
### Step 9
################################################################################

    df = pd.DataFrame(shortened, columns = ['text'])
    df['n_tokens'] = df.text.apply(lambda x: len(tokenizer.encode(x)))
    df.n_tokens.hist()

################################################################################
### Step 10
################################################################################


    df['embeddings'] = df.text.apply(lambda x: openai.Embedding.create(input=x, engine='text-embedding-ada-002')['data'][0]['embedding'])
    # store the df into Pinecone DB.
    
    # Add an 'id' column to the DataFrame
    df['id'] = [str(uuid4()) for _ in range(len(df))]

    # Define index name
    index_name = 'quickquacktest'
    nspace = extract_domain_name(domain=domain)


    # Initialize connection to Pinecone
    pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)

    # Check if index already exists, create it if it doesn't
    try:
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(index_name, dimension=1536, metric='dotproduct')
    except Exception as e:
        print(e)
  

    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    index.describe_index_stats()

    batch_size = 100  # how many embeddings we create and insert at once
    # Upsert ends up adding duplicate vectors instead of updating, So delete the exsisting namespace before adding new vectors.
    try:
        print("delete start")
        delete_response = index.delete(delete_all=True,namespace=nspace) # With Namespace.
        delete_response = index.delete(delete_all=True)
        print("delete end")
    except Exception as e:
        print(e)

    # Convert the DataFrame to a list of dictionaries
    text_chunks = df.to_dict(orient='records')


    # Upsert embeddings into Pinecone in batches of 100
    for i in tqdm(range(0, len(text_chunks), batch_size)):
        i_end = min(len(text_chunks), i+batch_size)
        meta_batch = text_chunks[i:i_end]
        ids_batch = [x['id'] for x in meta_batch]
        embeds = [x['embeddings'] for x in meta_batch]
        meta_batch = [{
            'text': x['text']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        index.upsert(vectors=to_upsert, namespace=nspace) # With Namespace.
        # index.upsert(vectors=to_upsert)
        print("upsert done")
    
    if os.path.exists("app/webcrawler/text/"):
        shutil.rmtree("app/webcrawler/text/")
    
    if os.path.exists("app/webcrawler/processed"):
        shutil.rmtree("app/webcrawler/processed")


################################################################################
### Step 11
################################################################################

    #df=pd.read_csv('app/webcrawler/processed/embeddings.csv', index_col=0)
    #df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    #return df

################################################################################
### Step 12
################################################################################

limit = 3750

def retrieve(query, domain):
    res = openai.Embedding.create(
        input=[query],
        engine='text-embedding-ada-002'
    )
    namespace = extract_domain_name(domain=domain)
    # Define index name
    index_name = 'quickquacktest'
    # Connect to the index and view index stats
    index = pinecone.Index(index_name)
    index.describe_index_stats()


    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True, namespace=namespace)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below. If you don't know the answer, just say that you don't know. Don't try to make up an answer.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )

    # Initialize prompt with a default value
    # prompt = "No context found"
    # append contexts until hitting limit
    for i in range(1, len(contexts)):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts[:i-1]) +
                prompt_end
            )
            break
        elif i == len(contexts)-1:
            prompt = (
                prompt_start +
                "\n\n---\n\n".join(contexts) +
                prompt_end
            )
    return prompt

    
def answer_question(question, domain):

    query_with_contexts = retrieve(question, domain)

    try:
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a Q&A assistant."},
                {"role": "user", "content": query_with_contexts}
            ]
        )
        return chat["choices"][0]['message']['content']
    except Exception as e:
        print(e)
        return ""

################################################################################
### Step 13
################################################################################

# print(answer_question(df, question="What coffee do you serve?", debug=False))

# print(answer_question(df, question="How much does a mocha cost?"))

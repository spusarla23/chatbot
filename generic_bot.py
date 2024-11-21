import os # os to access environ variables like API Key
import openai # to generate embeddings
from dotenv import load_dotenv # load env variables
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance # for collection setup
from uuid import uuid4 # to create unique IDs for documents

#load envir variables from .env file
load_dotenv()

#Retrieve the openAI API Key from env variable
api_key = os.getenv("OPENAI_API_KEY")
if api_key is None:
    print("Error: OPENAI_API_KEY not found in .env file")
else:
    print("API Key successfully loaded")

#Qdrant client setup
qdrant_client = QdrantClient(
    url=os.getenv("QDRANT_URL"),
    api_key=os.getenv("QDRANT_API_KEY")
)

# Function to create embeddings using OpenAI API
def create_embeddings(text: str):
    try:
        response = openai.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        
        create_embeddings = response.data[0].embedding
        return create_embeddings
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

# function to store the embeddings in Qdrant database
def store_embeddings_in_qdrant(collection_name: str, embeddings, metadata: dict):
    try:
        # convert embeddings and metadata to correct format
        points = [
            {
                'id': str(uuid4()), # Generating a unique ID for each document
                'vector': embeddings,
                'payload': metadata
            }
        ]

        #Create the collection if it dosent exist
        if collection_name not in [c.name for c in qdrant_client.get_collections().collections]:
            qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=len(embeddings), distance=Distance.COSINE)
            )
            print(f"collection '{collection_name}' created.")
        
        #Insert the embeddings and metadata into Qdrant
        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"Embeddings for collection '{collection_name}' added to Qdrant")

    except Exception as e:
        print(f"Error storing embeddings in Qdrant")

# function to retrieve embeddings based on users query
def retrieve_similar_documents(query: str, collection_name: str):
    query_embedding = create_embeddings(query)
    search_result = qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=3
    )
    return search_result

def run_chatbot():
    print("chatbot: Hellp! Type 'exit' to end the conversation.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'exit':
            print("Chatbot: GoodBye!")
            break
        response = retrieve_similar_documents(user_input, collection_name="openai_embeddings")
        print("Chatbot: Here's what i found: ")
        for result in response:
            print(f"- {result.payload['text']}")

#Test the embedding
if __name__ == "__main__":
   
    #sample_text = "This is a test sentence for generating embeddings."
    #print(sample_text)
    '''sample_text = """His picture had a surpassing influence over my life. As I grew, the thought of the master grew
with me. In meditation I would often see his photographic image emerge from its small frame
and, taking a living form, sit before me. When I attempted to touch the feet of his luminous body,
it would change and again become the picture. As childhood slipped into boyhood, I found Lahiri
Mahasaya transformed in my mind from a little image, cribbed in a frame, to a living,
enlightening presence. I frequently prayed to him in moments of trial or confusion, finding within
me his solacing direction. At first I grieved because he was no longer physically living. As I began
to discover his secret omnipresence, I lamented no more. He had often written to those of his
disciples who were over-anxious to see him: "Why come to view my bones and flesh, when I am
ever within range of your kutastha (spiritual sight)?"
I was blessed about the age of eight with a wonderful healing through the photograph of Lahiri
Mahasaya. This experience gave intensification to my love. While at our family estate in Ichapur,
Bengal, I was stricken with Asiatic cholera. My life was despaired of; the doctors could do
nothing. At my bedside, Mother frantically motioned me to look at Lahiri Mahasaya's picture on
the wall above my head"""
    embeddings = create_embeddings(sample_text)
    if embeddings:
        print(f"Embeddings for the text: {embeddings[:5]}") #show the first 5 embedding values

    #store embeddings in Qdrant if created successfully
    if embeddings:
        metadata = {"text": sample_text} #raw data
        store_embeddings_in_qdrant("openai_embeddings", embeddings, metadata)
        '''
    run_chatbot()
   
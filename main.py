import os
import json
import logging
import google.cloud.logging
from flask import Flask, render_template, request

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client()
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Application Variables
BOTNAME = "AutoQuest" 
SUBTITLE = "Especialista Amigável em Manuais de Veículos"

app = Flask(__name__)

# Initialize Firestore Client with Project ID
PROJECT_ID = "pesquisaia-454015" 
db = firestore.Client(project=PROJECT_ID)

# Firestore Collection Reference

collection = db.collection('cod-civil2')

# Vertex AI Embedding Model Initialization
embedding_model = VertexAIEmbeddings("text-multilingual-embedding-002")

# Vertex AI Generative Model Initialization
gen_model = GenerativeModel(model_name="gemini-1.5-pro-001")


# Function to Search the Vector Database and Retrieve Relevant Context
def search_vector_database(query: str):
    query_embedding = embedding_model.embed_query(query)
    vector_query = collection.find_nearest(
        vector_field="embedding",
        query_vector=Vector(query_embedding),
        distance_measure=DistanceMeasure.EUCLIDEAN,
        limit=5,
    )
    docs = vector_query.stream()
    context = " ".join([result.to_dict()['content'] for result in docs])

    # Log the context retrieved from the vector database
    logging.info(context, extra={"labels": {"service": "cymbal-service", "component": "context"}})
    return context

# Function to Send Query and Context to Gemini and Get the Response
def ask_gemini(question):
    # Create a prompt template with context instructions
    prompt_template = """
    Você é um especialista no assunto e deve responder com precisão à pergunta do usuário, utilizando **somente as informações do contexto** fornecido abaixo. 

    **Regras para a resposta**:
    - Se a resposta não estiver no contexto, responda **"Não encontrei informações suficientes no contexto para responder."**.
    - Se houver múltiplas informações no contexto, sintetize e apresente **de forma objetiva**.
    - **Evite suposições** e não invente informações.
    - Se necessário, cite a **fonte** da resposta.

    **Contexto**:
    {context}

    **Pergunta**: {question}

    **Resposta**:
        """
    # Retrieve context for the question using the search_vector_database function
    context = search_vector_database(question)
    
    # Format the prompt with the question and retrieved context
    formatted_prompt = prompt_template.format(context=context, question=question)
    
    # Define the generation configuration for the Gemini model
    generation_config = GenerationConfig(
        temperature=0.7,  # Adjust temperature as needed
        max_output_tokens=256,  # Maximum tokens in the response
        response_mime_type="application/json",  # MIME type of response
    )
    
    # Define the contents parameter with the prompt text
    contents = [
        {
            "role": "user",
            "parts": [{"text": formatted_prompt}]
        }
    ]
    
    # Call the generate_content function with the defined parameters
    response = gen_model.generate_content(
        contents=contents,
        generation_config=generation_config
    )
    
    # Parse the JSON response to extract the answer field
    response_text = response.text if response else "{}"  # Default to empty JSON if no response
    try:
        response_json = json.loads(response_text)  # Parse the JSON string into a dictionary
        answer = response_json.get("answer", "No answer found.")  # Get the "answer" field
    except json.JSONDecodeError:
        answer = "Error: Unable to parse response."

    return answer

# Home page route
@app.route("/", methods=["POST", "GET"])
def main():
    # Initial message for GET request
    if request.method == "GET":
        question = ""
        answer = "Olá, eu sou o AutoQuest, o que posso fazer por você?"

    # Handle POST request when the user submits a question
    else:
        question = request.form["input"]
        
        # Log the user's question
        logging.info(question, extra={"labels": {"service": "cymbal-service", "component": "question"}})
        
        # Get the answer from Gemini based on the vector database search
        answer = ask_gemini(question)

    # Log the generated answer
    logging.info(answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}})
    print("Answer: " + answer)

    # Display the home page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
    }

    return render_template("index.html", config=config)

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

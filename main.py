from flask import Flask, render_template, request, redirect, url_for
import os
import json
import logging
import google.cloud.logging

from google.cloud import firestore
from google.cloud.firestore_v1.vector import Vector
from google.cloud.firestore_v1.base_vector_query import DistanceMeasure

import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
from langchain_google_vertexai import VertexAIEmbeddings

# Configure Cloud Logging
logging_client = google.cloud.logging.Client(project="pesquisaia-454015")
logging_client.setup_logging()
logging.basicConfig(level=logging.INFO)

# Application Variables
BOTNAME = "AutoQuest" 
SUBTITLE = "Especialista Amigável em Manuais de Veículos"

app = Flask(__name__)

# Initialize Firestore Client with Project ID
PROJECT_ID = "pesquisaia-454015" 
db = firestore.Client(project=PROJECT_ID)

# Vertex AI Embedding Model Initialization
embedding_model = VertexAIEmbeddings(
    model_name="text-multilingual-embedding-002",
    project="pesquisaia-454015"
)

# Vertex AI Generative Model Initialization
gen_model = GenerativeModel(model_name="gemini-1.5-pro-001")

# Default configuration values
model_config = {
    "model_name": "gemini-1.5-pro-001",
    "temperature": 0.7,
    "max_tokens": 256,
    "mime_type": "application/json",
    "collection_name": "cod-civil2"
}

class busca:
    def __init__(self, nome, categoria, console):
        self.nome=nome
        self.categoria=categoria
        self.console=console
        

# Function to Search the Vector Database and Retrieve Relevant Context
def search_vector_database(query: str, collection_name: str):
    collection = db.collection(collection_name)
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
def ask_gemini(question, collection_name):
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
    context = search_vector_database(question, collection_name)
    
    # Format the prompt with the question and retrieved context
    formatted_prompt = prompt_template.format(context=context, question=question)
    
    # Define the generation configuration for the Gemini model
    generation_config = GenerationConfig(
        temperature=model_config["temperature"],  # Use the configured temperature
        max_output_tokens=model_config["max_tokens"],  # Use the configured max tokens
        response_mime_type=model_config["mime_type"],  # Use the configured MIME type
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
    
    # Print the raw response for debugging
    print("Raw response:", response.text)
    
    # Parse the JSON response to extract the answer field
    response_text = response.text if response else "{}"  # Default to empty JSON if no response
    try:
        response_json = json.loads(response_text)  # Parse the JSON string into a dictionary
        if isinstance(response_json, dict):
            answer = response_json.get("Resposta") or response_json.get("resposta") or response_json.get("answer", "No answer found.")  # Get the "Resposta", "resposta", or "answer" field
        else:
            answer = response_text  # If response_json is not a dict, use the raw response text
    except (json.JSONDecodeError, IndexError, KeyError):
        answer = "Error: Unable to parse response."

    return answer, context

# Home page route
@app.route("/", methods=["POST", "GET"])
def main():
    # Initial message for GET request
    if request.method == "GET":
        question = ""
        answer = "Olá, eu sou o AutoQuest, o que posso fazer por você?"
        context = ""

    # Handle POST request when the user submits a question
    else:
        question = request.form["input"]
        
        # Log the user's question
        logging.info(question, extra={"labels": {"service": "cymbal-service", "component": "question"}})
        
        # Get the answer from Gemini based on the vector database search
        answer, context = ask_gemini(question, model_config["collection_name"])

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
        "context": context
    }

    return render_template("index.html", config=config)

@app.route("/novo", methods=["POST", "GET"])
def novo():
    if request.method == "POST":
        # Update model configuration with the new values from the form
        model_config["model_name"] = request.form["model_name"]
        model_config["temperature"] = float(request.form["temperature"])
        model_config["max_tokens"] = int(request.form["max_tokens"])
        model_config["mime_type"] = request.form["mime_type"]
        model_config["collection_name"] = request.form["collection_name"]
        
        # Update the generative model with the new model name
        global gen_model
        gen_model = GenerativeModel(model_name=model_config["model_name"])
        
        return redirect(url_for("novo"))
    
    # Get all collections from Firestore
    collections = [collection.id for collection in db.collections()]
    
    return render_template('novo.html', collections=collections, **model_config)

@app.route("/chat", methods=["POST", "GET"])
def chat():
    if request.method == "GET":
        question = ""
        answer = "Olá, eu sou o AutoQuest, como posso ajudar você hoje?"
        context = ""
        chat_history = []

    else:
        question = request.form["input"]
        chat_history = request.form.get("chat_history", "[]")
        print("chat_history (raw):", chat_history)
        chat_history = json.loads(chat_history)
        print("chat_history (parsed):", chat_history)

        # Log the user's question
        logging.info(question, extra={"labels": {"service": "cymbal-service", "component": "question"}})

        # Get the answer from Gemini based on the vector database search
        answer, context = ask_gemini(question, model_config["collection_name"])

        # Add the question and answer to the chat history
        if isinstance(chat_history, list):
            chat_history.append({"question": question, "answer": answer})
        else:
            chat_history = [{"question": question, "answer": answer}]
        print("chat_history (updated):", chat_history)

    # Log the generated answer
    logging.info(answer, extra={"labels": {"service": "cymbal-service", "component": "answer"}})
    print("Answer: " + answer)

    # Display the chat page with the required variables set
    config = {
        "title": BOTNAME,
        "subtitle": SUBTITLE,
        "botname": BOTNAME,
        "message": answer,
        "input": question,
        "context": context,
        "chat_history": chat_history  # Pass the list directly
    }
    print("config:", config)

    return render_template("chat.html", config=config)

@app.route("/config", methods=["POST", "GET"])
def config():
    if request.method == "POST":
        # Update model configuration with the new values from the form
        model_config["model_name"] = request.form["model_name"]
        model_config["temperature"] = float(request.form["temperature"])
        model_config["max_tokens"] = int(request.form["max_tokens"])
        model_config["mime_type"] = request.form["mime_type"]
        model_config["collection_name"] = request.form["collection_name"]
        
        # Update the generative model with the new model name
        global gen_model
        gen_model = GenerativeModel(model_name=model_config["model_name"])
        
        return redirect(url_for("config"))
    
    # Get all collections from Firestore
    collections = [collection.id for collection in db.collections()]
    
    return render_template('config.html', collections=collections, **model_config)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

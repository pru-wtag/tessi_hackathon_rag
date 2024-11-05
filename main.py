from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.legacy.llms import Ollama
from ollama import Client
from llama_index.core import Settings

Settings.llm = Ollama(model='phi3', base_url='http://localhost:11434', temperature=0.1)


def load_documents_and_build_index(query):

    # Load documents from the specified dataset directory
    documents = SimpleDirectoryReader("dataset").load_data()

    # Build an index from the loaded documents
    index = VectorStoreIndex.from_documents(documents)

    # Create a query engine from the index
    query_engine = index.as_query_engine()

    # Retrieve and format query results from the engine
    raw_response = query_engine.query(query)
    processed_response = raw_response.get_response_text()

    # Use Ollama to refine or process the response
    ollama_response = process_with_ollama(processed_response)

    return ollama_response


def process_with_ollama(text):

    client = Client(host='http://localhost:11434')

    # Define the prompt for Ollama
    prompt = {
        'role': 'user',
        'content': text
    }

    # Send the prompt to Ollama and retrieve the response
    response = client.chat(model='llama3.1', messages=[prompt])

    return response['message']['content']


def evaluate_with_ollama(response_text):

    client = Client(host='http://localhost:11434')

    # Define the evaluation prompt
    evaluation_prompt = {
        'role': 'user',
        'content': f"Based on the following text, determine if the described service request is billable: \n\n{response_text}"
    }

    # Send the prompt to Ollama for evaluation
    response = client.chat(model='llama3.1', messages=[evaluation_prompt])

    return response['message']['content']


def main():
    """
    Main function to load documents, process a query, and evaluate the response.
    """
    # Sample customer request for evaluation
    customer_request = "Request for emergency maintenance service on IMBL Scanner due to operational failure."

    # Process the customer request using the query engine and Ollama
    response_text = load_documents_and_build_index(customer_request)

    # Evaluate the response text using Ollama's model
    evaluation_result = evaluate_with_ollama(response_text)

    # Display the results
    print("Customer Request:", customer_request)
    print("\nQuery Response:")
    print(response_text)
    print("\nEvaluation Result:")
    print(evaluation_result)


if __name__ == "__main__":
    main()

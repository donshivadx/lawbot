from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import pandas as pd
import os
import openai
from transformers import pipeline

app = Flask(__name__)
socketio = SocketIO(app)

# Set your OpenAI API key here
api_key = "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"
openai.api_key = api_key

# Load BERT question answering pipeline
bert_qa_pipeline = pipeline("question-answering", model="deepset/bert-base-cased-squad2")

def get_openai_response(prompt):
    response = openai.Completion.create(
        engine="text-davinci-002",
        prompt=prompt,
        max_tokens=1000  # Increase max_tokens to ensure longer responses
    )
    text = response.choices[0].text.strip()
    paragraphs = text.split("\n\n")  # Split by double line breaks to form paragraphs
    return paragraphs

def get_bert_response(question, context):
    result = bert_qa_pipeline({
        'question': question,
        'context': context
    })
    return result['answer']

def get_user_input(column_name):
    while True:
        user_input = input(f"Do you think this case has {column_name} (0 for No, 1 for Yes): ").strip()
        if user_input in {'0', '1'}:
            return 'No' if user_input == '0' else 'Yes'
        else:
            print("Invalid input, please enter 0 for No or 1 for Yes.")

def load_csv(file_path):
    print("File Path:", file_path)  # Add this line to print the file path
    return pd.read_csv(file_path)

def find_outcome(df, user_inputs):
    user_inputs_df = pd.DataFrame([user_inputs])
    matching_row = df[df[df.columns[:-1]].eq(user_inputs_df.iloc[0]).all(axis=1)]

    if not matching_row.empty:
        return matching_row['outcome'].values[0]
    else:
        return "Outcome not found for the given inputs."

def chat_with_bot(case_type):
    # Define the CSV file path based on the user's choice
    csv_file_path = f'{case_type.lower()}.csv'

    # Check if the CSV file exists
    if not os.path.exists(csv_file_path):
        yield {'message': f"CSV file not found for {case_type}. Please make sure the file exists."}

    # Load the CSV file into a DataFrame
    df = load_csv(csv_file_path)

    # Initialize user_inputs dictionary
    user_inputs = {}

    # Get user inputs for each column
    for column in df.columns[:-1]:
        user_inputs[column] = get_user_input(column)

        # Send user message to frontend
        yield {'user_message': f"Do you think this case has {column} (0 for No, 1 for Yes): {user_inputs[column]}"}

    # Find and print the outcome based on user inputs
    outcome = find_outcome(df, user_inputs)

    if outcome == "Outcome not found for the given inputs.":
        # If outcome not found in CSV, use OpenAI for a response
        user_inputs_str = ", ".join(f"{key}: {value}" for key, value in user_inputs.items())
        SHIVA = f"what will be the outcome of this case take {user_inputs_str}and can I file a case(charge on them) or sue on them and what are the documents needed and what is your suggestion for this case, and what are all the IPC sections I can file on them according to India"
        openai_response = get_openai_response(SHIVA)

        # Store GPT-3 response
        gpt_responses = openai_response

        # Send GPT-3 response to frontend
        yield {'bot_message': openai_response}

        # Continue the chat with GPT
        while True:
            user_input_gpt = input("You: ")
            if user_input_gpt.lower() == 'exit':
                break
            elif user_input_gpt.lower() == 'bert':
                # Switch to BERT
                while True:
                    user_input_bert = input("You (for BERT): ")
                    if user_input_bert.lower() == 'exit':
                        break
                    context_for_bert = " ".join(gpt_responses)  # Use GPT-3 responses as context for BERT
                    bert_response = get_bert_response(user_input_bert, context_for_bert)
                    print(f"BERT: {bert_response}")

                    # Send BERT response to frontend
                    yield {'bot_message': [bert_response]}
            else:
                # Call OpenAI API with user input
                prompt_gpt = f"You: {user_input_gpt}\nSHIVA:"
                response_gpt = get_openai_response(prompt_gpt)

                # Store GPT-3 response
                gpt_responses.extend(response_gpt)

                # Send GPT-3 response to frontend
                yield {'bot_message': response_gpt}
    else:
        # Print outcome and ask for further input
        yield {'outcome': outcome}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('start_chat')
def handle_start_chat(case_type):
    for message in chat_with_bot(case_type):
        emit('chat_message', message)

if __name__ == "__main__":
    socketio.run(app, debug=True)





































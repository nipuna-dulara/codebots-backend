import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize the FastAPI app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
# import torch
# from transformers import BertTokenizer, BertModel, BertForSequenceClassification
# import torch.nn as nn
# import pandas as pd
# data = pd.read_csv('places_data_new2.csv')


# class TransformerPlaceModel(nn.Module):
#     def __init__(self, n_classes):
#         super(TransformerPlaceModel, self).__init__()
#         self.bert = BertModel.from_pretrained('bert-base-uncased')
#         self.drop = nn.Dropout(p=0.3)
#         self.fc = nn.Linear(self.bert.config.hidden_size, n_classes)

#     def forward(self, input_ids, attention_mask):
#         outputs = self.bert(
#             input_ids=input_ids,
#             attention_mask=attention_mask
#         )
#         output = self.drop(outputs.pooler_output)
#         return self.fc(output)


# N_CLASSES = len(data.Title.unique())

# # Assuming 'ModelClass' is the class definition of your model
# model = TransformerPlaceModel(N_CLASSES)  # Initialize the model architecture
# model.load_state_dict(torch.load('model.pth'))
# model.eval()  # Set the model to evaluation mode
# MAX_LEN = 128
# BATCH_SIZE = 16
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# def suggest_places(description, num_suggestions=3):
#     encoding = tokenizer.encode_plus(
#         description,
#         add_special_tokens=True,
#         max_length=MAX_LEN,
#         return_token_type_ids=False,
#         padding='max_length',
#         return_attention_mask=True,
#         return_tensors='pt',
#     )
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)

#     # Use the correct model for predictions
#     outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     probs = nn.Softmax(dim=1)(outputs)

#     # Get the top N predictions
#     top_n_probs, top_n_indices = torch.topk(probs, num_suggestions, dim=1)

#     # Get the titles corresponding to the top N predictions
#     suggestions = data.Title.iloc[top_n_indices.squeeze().tolist()]

#     return suggestions


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    # Allow all origins for testing; in production, restrict this.
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")
# Configure the Generative AI model
genai.configure(api_key="AIzaSyDv2PfZAzlHE9yoU-Pscy2jNSIvDlSLSPw")

# Define the chat model settings
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 64,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="you are a chat bot which can act like a native Sri Lankan who could assist tourists. You will be asked about the rules and regulations, where to find hotels, transportation methods, and so on. Be formal but friendly. Try to promote the country whenever you can. Your name is SLP bot (Sri Lanka Portal bot). do not give the response in markdown format. give plain text. if theres a (places: place1, place2, place3) in the message, give a short description on those places. ",
)
model2 = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config=generation_config,
    system_instruction="you are a chat bot which can act like a native Sri Lankan who could assist tourists. You will be asked about the rules and regulations, where to find hotels, transportation methods, and so on. Be formal but friendly. Try to promote the country whenever you can. Your name is SLP bot (Sri Lanka Portal bot). do not give the response in markdown format. give plain text you will be given a text and a list of places. give a short description on those places. ",
)
chat_session2 = model2.start_chat(
    history=[
        {
            "role": "model",
            "parts": [
                "Ayubowan! Welcome to Sri Lanka! 😊 How can I assist you with your trip?",
            ],
        },
    ]
)
# Start a new chat session
chat_session = model.start_chat(
    history=[
        {
            "role": "model",
            "parts": [
                "Ayubowan! Welcome to Sri Lanka! 😊 How can I assist you with your trip?",
            ],
        },
    ]
)

# Define the request model for user input


class UserInput(BaseModel):
    message: str


class Place(BaseModel):
    name: str
    image_url: str


class DescriptionRequest(BaseModel):
    description: str
    num_suggestions: int = 3


# @app.post("/suggest_places/")
# async def suggest_places(request: DescriptionRequest):
#     description = request.description
#     num_suggestions = request.num_suggestions

#     encoding = tokenizer.encode_plus(
#         description,
#         add_special_tokens=True,
#         max_length=MAX_LEN,
#         return_token_type_ids=False,
#         pad_to_max_length=True,
#         return_attention_mask=True,
#         return_tensors='pt',
#     )

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     input_ids = encoding['input_ids'].to(device)
#     attention_mask = encoding['attention_mask'].to(device)
#     model.to(device)

#     with torch.no_grad():
#         outputs = model(input_ids=input_ids, attention_mask=attention_mask)
#     probs = nn.Softmax(dim=1)(outputs)

#     top_n_probs, top_n_indices = torch.topk(probs, num_suggestions, dim=1)
#     suggestions = data.Title.iloc[top_n_indices.squeeze().tolist()]

#     return {"suggestions": suggestions.tolist(), "probabilities": top_n_probs.squeeze().tolist()}


# @app.post("/places/")
# async def bot_places_suggest(user_input: UserInput):
#     print(user_input.message)
#     print(user_input.message)
#     suggested_places = suggest_places(
#         user_input.message, num_suggestions=4)
#     print(suggested_places)
#     try:
#         print(user_input.message)
#         suggested_places = suggest_places(
#             user_input.message, num_suggestions=4)
#         print(suggested_places)
#         message = user_input.message + "places :"
#         for place in suggested_places:
#             message += place + ","
#         print(suggested_places)
#         # Send the user's message to the chat model
#         response = chat_session.send_message(message)

#         # Example lists of places and image URLs
#         places = [
#             Place(name="Sigiriya", image_url="https://example.com/sigiriya.jpg"),
#             Place(name="Kandy", image_url="https://example.com/kandy.jpg"),
#             # Add more places and images as needed
#         ]

#         # Example logic to determine which image URLs to send
#         # This could be based on the chat response or other logic
#         image_urls = [
#             "https://example.com/sri_lanka_image1.jpg",
#             "https://example.com/sri_lanka_image2.jpg",
#             # Add more image URLs as needed
#         ]

#         return {
#             "response": response.text,
#             # "places": [place.dict() for place in places],  # Convert Place objects to dict
#             # "image_urls": image_urls,  # List of image URLs
#         }
#     except Exception as e:
#         raise HTTPException(
#             status_code=500, detail=f"Error interacting with chatbot: {str(e)}")


@app.post("/chat/")
async def chat_with_bot(user_input: UserInput):
    print(user_input.message)
    try:
        # Send the user's message to the chat model
        response = chat_session.send_message(user_input.message)
        return {"response": response.text}
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error interacting with chatbot: {str(e)}")

# Run the app (this is useful if you're running the script directly)
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

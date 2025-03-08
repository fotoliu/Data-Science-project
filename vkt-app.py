from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import random
import praw
from reddit import fetch_reddit_user
from transformers import BertTokenizer, BertModel
import torch
from training2 import BotDetectionModel, preprocess_user_data

app = FastAPI()
templates = Jinja2Templates(directory="templates")

def predict_single_entry(model, tokenizer, user_data, device='cuda' if torch.cuda.is_available() else 'cpu'):
    model.to(device)
    model.eval()  # Set model to evaluation mode

    # Preprocess user data
    text, numerical_features = preprocess_user_data(user_data)

    # Tokenize text
    encoding = tokenizer(
        text,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )

    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    numerical_features = numerical_features.unsqueeze(0).to(device)  # Add batch dimension

    # Make prediction
    with torch.no_grad():
        output = model(input_ids, attention_mask, numerical_features)
        probability = output.item()  # Get probability score

    return probability

def predict_reddit_user(username, model, tokenizer, device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Fetch user data
    user_data = fetch_reddit_user(username)
    
    # Handle errors in fetching
    if "Error" in user_data:
        print(f"Error fetching user: {user_data['Error']}")
        return None
    
    # Predict bot probability
    bot_probability = predict_single_entry(model, tokenizer, user_data, device)
    
    return bot_probability

# Load trained model
bert_model = BertModel.from_pretrained('bert-base-uncased')
model = BotDetectionModel(bert_model)
model.load_state_dict(torch.load('bot_detection_model.pth', map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')))
model.eval()

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')





@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/test", response_class=HTMLResponse)
async def test_user(request: Request, username: str = Form(...)):

    probability = predict_reddit_user(username, model, tokenizer)
    
    classification = "Bot" if probability > 0.5 else "Human"
    return templates.TemplateResponse(
        "result.html",
        {"request": request, "username": username, "classification": classification, "probability": probability},
    )
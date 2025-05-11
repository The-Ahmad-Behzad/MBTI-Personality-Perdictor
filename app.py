from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import re
import json

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Model paths
MODEL_PATH = './models/mbti_bert_best_model.pt'
TOKENIZER_PATH = './models/tokenizer'
CONFIG_PATH = './models/model_config.json'

# Load model configuration
with open(CONFIG_PATH, 'r') as f:
    config = json.load(f)

MAX_LEN = config['max_len']
label_dict = config['label_dict']
reverse_label_dict = config['reverse_label_dict']

# Convert string keys back to integers for reverse_label_dict
reverse_label_dict = {int(k): v for k, v in reverse_label_dict.items()}

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained(TOKENIZER_PATH)

# Initialize the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label_dict),
    output_attentions=False,
    output_hidden_states=False
)

# Load the saved model state
checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

# Text preprocessing function
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    
    # Remove emojis and special characters
    text = re.sub(r'[^\w\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def predict_mbti_type(texts):
    """
    Predict MBTI type for a list of up to 3 text samples.
    """
    # Ensure we have at most 3 texts
    texts = texts[:3]
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Join texts with [SEP] token
    combined_text = ' [SEP] '.join(processed_texts)
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        combined_text,
        add_special_tokens=True,
        max_length=MAX_LEN,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    
    # Move to device
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
    
    # Get predicted class
    predicted_class = torch.argmax(logits, dim=1).item()
    
    # Convert to MBTI type
    predicted_mbti_type = reverse_label_dict[predicted_class]
    
    # Get probability distribution
    probabilities = torch.nn.functional.softmax(logits, dim=1)[0].cpu().numpy()
    probs_dict = {mbti_type: float(probabilities[i]) for mbti_type, i in label_dict.items()}
    
    return predicted_mbti_type, probs_dict

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()
        
        # Extract texts
        texts = data.get('texts', [])
        
        # Ensure we have at least one text
        if not texts or len(texts) == 0:
            return jsonify({
                'error': 'Please provide at least one text sample.'
            }), 400
        
        # Make prediction
        predicted_type, probabilities = predict_mbti_type(texts)
        
        # Sort probabilities
        sorted_probs = {k: v for k, v in sorted(probabilities.items(), key=lambda item: item[1], reverse=True)}
        
        # Return prediction
        return jsonify({
            'predicted_type': predicted_type,
            'probabilities': sorted_probs
        })
    
    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
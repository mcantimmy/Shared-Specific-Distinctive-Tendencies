import streamlit as st
import torch
import torch.nn as nn
from transformers import RobertaTokenizer, RobertaModel

class ContrastiveAuthorshipModel(nn.Module):
    def __init__(self, pretrained_model_name='roberta-base'):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained_model_name)
        self.projection = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids1, attention_mask1, input_ids2, attention_mask2):
        output1 = self.roberta(input_ids1, attention_mask=attention_mask1)
        output2 = self.roberta(input_ids2, attention_mask=attention_mask2)
        
        embedding1 = self.projection(self.dropout(output1.last_hidden_state[:, 0, :]))
        embedding2 = self.projection(self.dropout(output2.last_hidden_state[:, 0, :]))
        
        return embedding1, embedding2

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ContrastiveAuthorshipModel().to(device)
    model.load_state_dict(torch.load('best_authorship_model.pth', map_location=device))
    model.eval()
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    return model, tokenizer, device

def predict_authorship(model, tokenizer, text1, text2, device, threshold=0.5):
    model.eval()
    encoding1 = tokenizer.encode_plus(
        text1,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    encoding2 = tokenizer.encode_plus(
        text2,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids1 = encoding1['input_ids'].to(device)
    attention_mask1 = encoding1['attention_mask'].to(device)
    input_ids2 = encoding2['input_ids'].to(device)
    attention_mask2 = encoding2['attention_mask'].to(device)
    
    with torch.no_grad():
        embedding1, embedding2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
        similarity = nn.functional.cosine_similarity(embedding1, embedding2).item()
    
    same_author = similarity > threshold
    return same_author, similarity

st.title('Authorship Verification App')

st.write("""
This app uses a deep learning model to predict whether two texts are written by the same author.
Enter two texts below and click 'Predict' to see the results.
""")

text1 = st.text_area("Enter the first text:", height=150)
text2 = st.text_area("Enter the second text:", height=150)

if st.button('Predict'):
    if text1 and text2:
        model, tokenizer, device = load_model()
        same_author, similarity = predict_authorship(model, tokenizer, text1, text2, device)
        
        st.write(f"Similarity score: {similarity*100:.2f}%")
        
        if same_author:
            st.success("The texts are likely written by the same author.")
        else:
            st.error("The texts are likely written by different authors.")
        
        st.write(f"Confidence: {abs(similarity-0.5)*200:.2f}%")
    else:
        st.warning("Please enter both texts before predicting.")

st.write("""
Note: This model's predictions are based on learned patterns and may not always be accurate. 
Factors such as text length, topic, and writing context can affect the results.
""")
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import RobertaTokenizer, RobertaModel
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from sklearn.model_selection import KFold
import numpy as np
from tqdm import tqdm

class RedditAuthorshipDataset(Dataset):
    def __init__(self, texts, authors, tokenizer, max_length=128):
        self.texts = texts
        self.authors = authors
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text1, author1 = self.texts[idx], self.authors[idx]
        # Randomly select another sample
        other_idx = np.random.randint(len(self.texts))
        text2, author2 = self.texts[other_idx], self.authors[other_idx]
        
        label = 1 if author1 == author2 else 0

        encoding1 = self.tokenizer.encode_plus(
            text1,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        encoding2 = self.tokenizer.encode_plus(
            text2,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids1': encoding1['input_ids'].flatten(),
            'attention_mask1': encoding1['attention_mask'].flatten(),
            'input_ids2': encoding2['input_ids'].flatten(),
            'attention_mask2': encoding2['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

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


def contrastive_loss(embedding1, embedding2, label, temperature=0.5):
    cosine_similarity = nn.functional.cosine_similarity(embedding1, embedding2)
    loss = torch.mean((1 - label) * torch.pow(cosine_similarity, 2) +
                      label * torch.pow(torch.clamp(1 - cosine_similarity, min=0.0), 2))
    return loss

def train(model, train_loader, val_loader, optimizer, device, epochs=5, patience=3):
    model.train()
    best_val_loss = float('inf')
    early_stopping_counter = 0
    
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)

            optimizer.zero_grad()
            embedding1, embedding2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = contrastive_loss(embedding1, embedding2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}")
        
        # Validation step
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'best_authorship_model.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    return model

def evaluate(model, val_loader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            input_ids1 = batch['input_ids1'].to(device)
            attention_mask1 = batch['attention_mask1'].to(device)
            input_ids2 = batch['input_ids2'].to(device)
            attention_mask2 = batch['attention_mask2'].to(device)
            labels = batch['label'].to(device)

            embedding1, embedding2 = model(input_ids1, attention_mask1, input_ids2, attention_mask2)
            loss = contrastive_loss(embedding1, embedding2, labels)
            total_loss += loss.item()

            similarity = nn.functional.cosine_similarity(embedding1, embedding2)
            predictions = (similarity > 0.5).long()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

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

def main():
    # Load Reddit dataset
    data = load_dataset("reddit", split="train[:100000]", trust_remote_code=True)
    texts = data['content']
    authors = data['author']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    dataset = RedditAuthorshipDataset(texts, authors, tokenizer)

    # K-Fold Cross-Validation
    n_splits = 2
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset), 1):
        print(f"Fold {fold}/{n_splits}")

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
        val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)

        train_loader = DataLoader(dataset, batch_size=32, sampler=train_subsampler)
        val_loader = DataLoader(dataset, batch_size=32, sampler=val_subsampler)

        model = ContrastiveAuthorshipModel().to(device)
        optimizer = optim.Adam(model.parameters(), lr=2e-5, weight_decay=0.01)

        model = train(model, train_loader, val_loader, optimizer, device, epochs=3, patience=2)

    # Load the best model for prediction
    best_model = ContrastiveAuthorshipModel().to(device)
    best_model.load_state_dict(torch.load('best_authorship_model.pth'))

    # Example prediction
    text1 = """There were no Dark Ages! They didn't happen! Byzantium was happily being Byzantium. The Muslims were doing fucking amazing things! Al-Andalus was a beacon of cultural integration, art, science, and philosophy! Ibn Khaldun was inventing modern history! 
            The 'Dark Ages' where when a bunch of dirt sucking savages from east-bumfuck lost contact with the First World, which is to say the Mediterranean. It was 'Dark' because no one who mattered gave two shits what was happening in Germania because Germania was utterly irrelevant to the world economy, sciences, history, and politics. Europe went through the 'Dark Ages' because Europe was not important. It was a worthless, cold, savage back water full of dirty hairy people who wore pants. 
            Right up until 700ish, when the Scandanavians went a viking and started to spread their culture across Northern Europe, setting up trade across the continent, forcing other NE cultures to centralize and become more efficient to resist the north men. 
            Seriously, though, the Muslims were rocking out with their Qu'ran out after about 600, and they did more for art, science, philosophy, and poetry than the Romans had done since 100ad. The period of Muslim ascendancy flowed smoothly out of the fall of Western Rome and then snugged seamlessly into the Renaissance. 
            And then the Norse were doing all sorts of wacky stuff with democracy and law from the mid millenium. Really, if there was a 'Dark Age' it was only from about 450, when the Romans abandoned Italy, to abou 600, when the Muslims really started kicking ass and taking names. 
            The only thing that was really 'lost' with the fall of Western Rome was the extremely powerful and centralized Roman state. All the cool technology they had persisted in other places (Specifically, everywhere except Europe), but without the massive centralization that let the Romans make use of it on such a large scale."""
    text2 = """I would associate the decline of the church largely to the loss of power of the Roman Empire in germania and western Europe, which was due to a large number of complicated factors, including over extension of the Empire's resources, migration of 'barbarian' peoples into the empire, the conflict between Pagan Roman religion and Christianity (Fun fact, the Visigoths that sacked Rome were Arian Christians, followers of a creed that had been declared heretical at the council of Nicea), and many, many other things. The Roman Empire was extremely important to pre-medieval Europe, introducing all kinds of culture and technology. When the financial and military support of the Empire withdrew much of that culture and technology went with it. 
            Also, I would like to note that up until... hmm, probably the 1500s or 1600s many, many powerful political figures were members were both Clergy and princes. Many Bishops and other church figures held land, raised armies, went to war, and participated in the councils of kings. They fought with secular lords and also with each other. 
            I would not say that the Church caused any decline in Europe, on the grounds that in many way there is no Europe without the Church and their is no Church without Europe. Catholicism was the culture of Europe from around 500 to around 1700. The Church was as important and basic a component of culture at that time as the Internet is now. Priests were often the only people with a semblance of education, the only people able to write and receive letters. While some theologians certainly advocated a radical and oppressive form of Christianity, others provided council to their leaders that served to limit the gross abuses of Feudalism. 
            In the end it's far too complicated to say that the Christian Church was a good thing or a bad thing. It spurred Europe to destructive wars with the Muslims to the south and the pagan Slavs in the east. It provided the foundations for rational inquiry on which Science was founded. It founded and promulgated the Inquisition, which was both a machine of torture and oppression and an instrument of social and political justice. The church preserved knowledge from the time of Rome and suppressed new knowledge. The church contributed and obstructed philosophy. 
            If Catholicism hadn't become the dominant religion in Europe I don't know that things would have changed very much. Certainly a Europe that followed the Mithras cult or kept to Roman or Germanic Paganism would be different, but I don't think humanity would necessarily have made more social or technical progress. The Romans could be as brutal and sadistic as any Inquistor, and the Vikings were notorious for being savage in battle. The Muslims put whole cities to the sword, and the Mongols carved a swath across the entire world. If Roman Catholicism hadn't risen to become the dominant cultural framework of Europe then it seems likely to me that one of those four groups, the Muslims, the Norsemen, the Romans, or the Mongols, would have shaped the face of Europe. Each culture had its great triumphs and terrible deeds."""
    same_author, similarity = predict_authorship(best_model, tokenizer, text1, text2, device)
    print(f"Texts are by the same author: {same_author}")
    print(f"Similarity score: {similarity:.4f}")

if __name__ == "__main__":
    main()
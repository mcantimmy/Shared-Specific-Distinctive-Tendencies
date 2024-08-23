import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, RobertaModel

# Set random seed for reproducibility
torch.manual_seed(42)

# Define the BERT-based Siamese Network
class RobertaSiameseNetwork(nn.Module):
    def __init__(self, roberta_model='roberta-base'):
        super(RobertaSiameseNetwork, self).__init__()
        self.roberta = RobertaModel.from_pretrained(roberta_model)
        self.fc = nn.Sequential(
            nn.Linear(self.roberta.config.hidden_size * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward_once(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :] 

    def forward(self, input1, input2):
        output1 = self.forward_once(input1['input_ids'], input1['attention_mask'])
        output2 = self.forward_once(input2['input_ids'], input2['attention_mask'])
        concat = torch.cat((output1, output2), 1)
        similarity = self.fc(concat)
        return similarity

# Custom Dataset
class AuthorshipDataset(Dataset):
    def __init__(self, text_pairs, labels, tokenizer, max_length=128):
        self.text_pairs = text_pairs
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text1, text2 = self.text_pairs[idx]
        label = self.labels[idx]
        
        encoding1 = self.tokenizer(text1, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        encoding2 = self.tokenizer(text2, padding='max_length', truncation=True, max_length=self.max_length, return_tensors='pt')
        
        return {
            'input1': {key: val.squeeze(0) for key, val in encoding1.items()},
            'input2': {key: val.squeeze(0) for key, val in encoding2.items()},
            'label': torch.tensor(label, dtype=torch.float32)
        }

# Data loading and preprocessing
def load_data(num_authors=1000, samples_per_author=2):
    data = load_dataset("reddit", split="train[:100000]", trust_remote_code=True)  # Limiting to 100k samples for this example
    
    author_groups = data.to_pandas().groupby('author')
    top_authors = author_groups.size().nlargest(num_authors).index.tolist()
    
    texts, authors = [], []
    for author in top_authors:
        author_texts = author_groups.get_group(author)['content'].tolist()
        if len(author_texts) >= samples_per_author:
            sampled_texts = np.random.choice(author_texts, samples_per_author, replace=False)
            texts.extend([str(text) for text in sampled_texts])
            authors.extend([author] * samples_per_author)
    
    return texts, authors

def create_pairs(texts, authors):
    pairs = []
    labels = []
    
    # Create positive pairs (same author)
    for i in range(0, len(texts), 2):
        pairs.append((texts[i], texts[i+1]))
        labels.append(1)
    
    # Create negative pairs (different authors)
    num_positive_pairs = len(pairs)
    author_indices = list(range(len(texts)))
    while len(pairs) < 2 * num_positive_pairs:
        idx1, idx2 = random.sample(author_indices, 2)
        if authors[idx1] != authors[idx2]:
            pairs.append((texts[idx1], texts[idx2]))
            labels.append(0)
    
    return pairs, labels

# Training function
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            input1, input2, labels = batch['input1'], batch['input2'], batch['label']
            input1 = {k: v.to(device) for k, v in input1.items()}
            input2 = {k: v.to(device) for k, v in input2.items()}
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(input1, input2)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                input1, input2, labels = batch['input1'], batch['input2'], batch['label']
                input1 = {k: v.to(device) for k, v in input1.items()}
                input2 = {k: v.to(device) for k, v in input2.items()}
                labels = labels.to(device)
                
                outputs = model(input1, input2)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()
                
                predicted = (outputs.squeeze() > 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        accuracy = 100 * correct / total
        
        print(f"Epoch {epoch+1}/{num_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {accuracy:.2f}%")
        print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

def main():
    print("Loading data...")
    texts, authors = load_data()
    
    print("Creating pairs...")
    pairs, labels = create_pairs(texts, authors)
    
    print("Initializing tokenizer...")
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    
    print("Preparing dataset...")
    X_train, X_test, y_train, y_test = train_test_split(pairs, labels, test_size=0.2, random_state=42)
    train_dataset = AuthorshipDataset(X_train, y_train, tokenizer)
    test_dataset = AuthorshipDataset(X_test, y_test, tokenizer)
    
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    print("Initializing model...")
    model = RobertaSiameseNetwork()
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    
    # Calculate total number of training steps
    num_epochs = 3
    total_steps = len(train_loader) * num_epochs
    
    # Create the learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,  # Default to 0, can be adjusted
        num_training_steps=total_steps
    )
    
    print("Starting training...")
    train(model, train_loader, test_loader, criterion, optimizer, scheduler, num_epochs=num_epochs)
    
    # Example verification
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
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
    encoding1 = tokenizer(text1, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    encoding2 = tokenizer(text2, padding='max_length', truncation=True, max_length=128, return_tensors='pt')
    input1 = {k: v.to(device) for k, v in encoding1.items()}
    input2 = {k: v.to(device) for k, v in encoding2.items()}
    
    with torch.no_grad():
        similarity = model(input1, input2)
    
    print(f"\nSample texts:")
    print(f"Text 1: '{text1}'")
    print(f"Text 2: '{text2}'")
    print(f"Similarity score: {similarity.item():.4f}")
    print(f"Prediction: {'Same author' if similarity.item() > 0.5 else 'Different authors'}")

if __name__ == "__main__":
    main()
# Fine-tuning a QA model on the crawled text
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")

# Load the data from the crawled text files
data = []
for file in os.listdir():
    if file.endswith(".txt"):
        with open(file, 'r', encoding='utf-8') as f:
            text = f.read()
            data.append(text)
            
# Encode the data using the tokenizer
encoded_data = tokenizer.batch_encode_plus(data, padding=True, truncation=True)

# Convert the encoded data into PyTorch tensors
input_ids = torch.tensor(encoded_data['input_ids'])
attention_mask = torch.tensor(encoded_data['attention_mask'])

# Fine-tune the model on the encoded data
model.train()
optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-5)
for epoch in range(3):
    for i in range(len(data)):
        # Set the input and output for the current data point
        input_dict = {
            'input_ids': input_ids[i].unsqueeze(0),
            'attention_mask': attention_mask[i].unsqueeze(0)
        }
        output_dict = tokenizer.encode_plus(
            "what is your question?",
            padding='max_length',
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )

        # Perform the forward pass and calculate the loss
        outputs = model(**input_dict, **output_dict)
        loss = outputs.loss

        # Backpropagate the loss and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save the fine-tuned model
model.save_pretrained('fine_tuned_model')
tokenizer.save_pretrained('fine_tuned_model')
print("Fine-tuned model saved!")
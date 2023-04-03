from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Load the fine-tuned tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model")
model = AutoModelForQuestionAnswering.from_pretrained("fine_tuned_model")

# List of questions to ask the model
questions = [
    "What is the capital of Tunisia?",
    "What is the official language of Tunisia?",
    "Who is the current president of Tunisia?",
    "When did Tunisia gain independence from France?",
    "What is the structure of the Tunisian government?"
]

# Loop through each question and ask the model
for question in questions:
    # Tokenize the input
    inputs = tokenizer(question, padding='max_length', truncation=True, max_length=64, return_tensors='pt')

    # Perform the forward pass
    outputs = model(**inputs)

    # Extract the start and end indices of the answer
    answer_start_scores = outputs.start_logits
    answer_end_scores = outputs.end_logits
    answer_start = torch.argmax(answer_start_scores)
    answer_end = torch.argmax(answer_end_scores) + 1

    # Decode the answer and print it
    answer_tokens = inputs['input_ids'][0][answer_start:answer_end]
    answer = tokenizer.decode(answer_tokens)
    print(f"Question: {question}")
    print(f"Answer: {answer}")

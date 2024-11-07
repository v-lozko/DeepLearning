#Valery Lozko
#CPSC 8430 
#HW3

from transformers import BertTokenizerFast, AutoModelForQuestionAnswering
from transformers import default_data_collator
from transformers import AdamW
from evaluate import load
import torch
import json
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#open files for train and test
with open("spoken_train-v1.1.json", "r") as f:
    train_data = json.load(f)

with open("spoken_test-v1.1.json", "r") as f:
    test_data = json.load(f)

#using BERT fast tokenizer with bert-large-uncased model
tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-large-uncased")
model = AutoModelForQuestionAnswering.from_pretrained("google-bert/bert-large-uncased")
model.to(device)

#flatten the data from train/test to be able to feed it into Dataset
def flatten_data(data):
    data_flat = []
    id_counter = 0
    for entry in data["data"]:
        for paragraph in entry["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                question = qa["question"]
                if qa["answers"]:
                    #only using the first answer for simplicity, though there could be multiple
                    first_answer = qa["answers"][0]
                else:
                    first_answer = {"answer_start": 0, "text": ""}
                data_flat.append({
                    "id": str(id_counter),
                    "context": context,
                    "question": question,
                    "answers": first_answer
                })
                id_counter += 1
    return data_flat

#function to tokenize data and find start and end positions of
# answers after tokenization using offset mapping
def preprocess_data_with_offsets(examples):
    inputs = tokenizer(
        examples["question"],
        examples["context"],
        truncation=True,
        padding="max_length",
        max_length=512,
        return_offsets_mapping=True
    )

    start_positions = []
    end_positions = []

    for i, answer in enumerate(examples["answers"]):
        start_char = answer["answer_start"]
        end_char = start_char + len(answer["text"])

        offsets = inputs["offset_mapping"][i]

        start_token = None
        end_token = None
        for j, (start, end) in enumerate(offsets):
            if start == start_char:
                start_token = j
            if end == end_char:
                end_token = j

        start_positions.append(start_token if start_token is not None else 0)
        end_positions.append(end_token if end_token is not None else 0)

    inputs.pop("offset_mapping")

    inputs.update({"start_positions": start_positions, "end_positions": end_positions})
    return inputs

#create dataset from training and testing data after flattening
train_data = Dataset.from_list(flatten_data(train_data))
test_data = Dataset.from_list(flatten_data(test_data))

#tokenizes the data via preprocess function
tokenized_train_data = train_data.map(preprocess_data_with_offsets, batched=True)
tokenized_test_data = test_data.map(preprocess_data_with_offsets, batched=True)

#prepare data for Dataloader by turning the columns into pytorch tensors
tokenized_train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions', 'id'])
tokenized_test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions', 'id'])

#load data into batches with dataloader
train_dataloader = DataLoader(
    tokenized_train_data, batch_size=8, shuffle=True, collate_fn=default_data_collator
)
test_dataloader = DataLoader(
    tokenized_test_data, batch_size=8, shuffle=True, collate_fn=default_data_collator
)

#optimizer
optimizer = AdamW(model.parameters(), lr=2e-5)

#mixed-precision scaler
scaler = GradScaler()

#number of epochs to train
num_epochs = 1

#metric to use to calculate F1 score
metric = load("squad_v2")

#function to evaluate teh model and calculate F1 score
def evaluate(model, dataloader, tokenizer):
    model.eval()
    predictions, references = [], []

    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            start_logits = outputs.start_logits
            end_logits = outputs.end_logits

            for i in range(len(input_ids)):
                #predicted start and end token positions
                start_pred = torch.argmax(start_logits[i]).item()
                end_pred = torch.argmax(end_logits[i]).item()

                #predicted answer span
                predicted_answer = tokenizer.decode(
                    input_ids[i][start_pred:end_pred + 1],
                    skip_special_tokens=True
                )

                #true answer span and start position
                true_start = batch["start_positions"][i].item()
                true_end = batch["end_positions"][i].item()
                true_answer = tokenizer.decode(
                    input_ids[i][true_start:true_end + 1],
                    skip_special_tokens=True
                )

                #add predictions
                predictions.append({
                    "id": str(i),
                    "prediction_text": predicted_answer,
                    "no_answer_probability": 0.0
                })

                #include both text and answer_start in references
                references.append({
                    "id": str(i),
                    "answers": [{"text": true_answer, "answer_start": true_start}]
                })

    # Compute metrics
    result = metric.compute(predictions=predictions, references=references)
    print(f"F1 Score: {result['f1']}, Exact Match: {result['exact']}")
    return result['f1'], result['exact']

#fine tune the model
for epoch in range(num_epochs):
    model.train()
    total_loss = 0

    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"):  # Progress bar for each epoch
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        start_positions = batch["start_positions"].to(device)
        end_positions = batch["end_positions"].to(device)

        optimizer.zero_grad()

        #mixed-precision training
        with autocast():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                start_positions=start_positions,
                end_positions=end_positions
            )
            loss = outputs.loss

        #scale the loss for FP16 precision
        scaler.scale(loss).backward()

        #step the optimizer with scaled gradients
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    #print the loss for information
    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss}")

#actually evaluate the model
evaluate(model, test_dataloader, tokenizer)
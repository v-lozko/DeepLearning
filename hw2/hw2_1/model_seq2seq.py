import sys
import torch
import torch.nn as nn
import numpy as np
import os
import json
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import random
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    #determine vocab for train and test based on train
    with open("training_label.json") as f:
        caption_data = json.load(f)

    all_captions = []
    for item in caption_data:
        all_captions.extend(item['caption'])

    word_counter = Counter()
    for caption in all_captions:
        tokens = caption.lower().split()
        word_counter.update(tokens)

    vocab = [word for word, freq in word_counter.items() if freq >= 1]
    other_tokens = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
    vocab = other_tokens + vocab

    # word indexes
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    idx2word = {idx: word for word, idx in word2idx.items()}

    # encoder model
    class Encoder(nn.Module):
        def __init__(self, input_dim, hidden_dim, num_layers, device):
            super(Encoder, self).__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
            self.device = device

        def forward(self, video_features):
            outputs, (hidden, cell) = self.lstm(video_features)
            return outputs, hidden, cell

    # attention model
    class Attention(nn.Module):
        def __init__(self, hidden_dim, device):
            super(Attention, self).__init__()
            self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.v = nn.Linear(hidden_dim, 1, bias=False)
            self.device = device

        def forward(self, hidden, encoder_outputs):
            hidden = hidden.unsqueeze(1)
            combined = torch.cat((hidden.expand(-1, encoder_outputs.size(1), -1), encoder_outputs), dim=2)
            energy = torch.tanh(self.attn(combined))
            attention = self.v(energy).squeeze(2)
            attention_weights = torch.softmax(attention, dim=1)
            context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)
            return context.squeeze(1), attention_weights

    # decoder model
    class Decoder(nn.Module):
        def __init__(self, output_dim, hidden_dim, num_layers, attention, device):
            super(Decoder, self).__init__()
            self.embedding = nn.Embedding(output_dim, hidden_dim)
            self.lstm = nn.LSTM(hidden_dim * 2, hidden_dim, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_dim * 2, output_dim)
            self.attention = attention
            self.device = device

        def forward(self, input_word, hidden, cell, encoder_outputs):
            embedded = self.embedding(input_word).unsqueeze(1)
            embedded = embedded.squeeze(2) if embedded.dim() == 4 else embedded
            context, _ = self.attention(hidden[-1], encoder_outputs)
            input = torch.cat((embedded, context.unsqueeze(1)), dim=2)
            output, (hidden, cell) = self.lstm(input, (hidden, cell))
            prediction = self.fc(torch.cat((output.squeeze(1), context), dim=1))
            return prediction, hidden, cell

    # seq2seq model
    class Seq2Seq(nn.Module):
        def __init__(self, encoder, decoder, device):
            super(Seq2Seq, self).__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.device = device

        def forward(self, video_features, target_captions, forcing_ratio=0.5):
            batch_size = target_captions.size(0)
            max_len = target_captions.size(1)
            vocab_size = self.decoder.fc.out_features
            outputs = torch.zeros(batch_size, max_len, vocab_size).to(self.device)
            encoder_outputs, hidden, cell = self.encoder(video_features)
            input_word = target_captions[:, 0]

            for t in range(1, max_len):
                prediction, hidden, cell = self.decoder(input_word, hidden, cell, encoder_outputs)
                outputs[:, t, :] = prediction
                top_prediction = prediction.argmax(dim=-1)
                use_forcing = random.random() < forcing_ratio
                input_word = target_captions[:, t] if use_forcing else top_prediction
            return outputs

    #variables for the network
    input_dim = 4096
    output_dim = len(word2idx)
    hidden_dim = 256
    num_layers = 2
    #i added .to(device) despite initializing to device because wasn't going to GPU
    encoder = Encoder(input_dim, hidden_dim, num_layers, device)
    encoder = encoder.to(device)
    attention = Attention(hidden_dim, device)
    attention = attention.to(device)
    decoder = Decoder(output_dim, hidden_dim, num_layers, attention, device)
    decoder = decoder.to(device)
    seq2seq = Seq2Seq(encoder, decoder, device)
    seq2seq = seq2seq.to(device)

    #if no arguments passed in will run in training mode
    if len(sys.argv) == 1:

        #function to load video features for training
        def load_video_features(directory):
            video_features = []
            for filename in os.listdir(directory):
                if filename.endswith(".npy"):
                    video_id = os.path.splitext(filename)[0]
                    features_path = os.path.join(directory, filename)
                    video_features.append((video_id,features_path))
            return video_features

        #function to match labels for each vid id
        def match_labels(video_features, labels_directory):
            with open(labels_directory, 'r') as f:
                caption_data = json.load(f)
            caption_dict = {item['id']: item['caption'] for item in caption_data}
            matched_data = []
            for video_id, features in video_features:
                if video_id in caption_dict:
                    captions = caption_dict[video_id]
                    matched_data.append((video_id, features, captions))
            return matched_data

        #restructure data for dataset
        def dataset_prep(matched_data):
            video_features = []
            caption_list = []
            for video_id, features, captions in matched_data:
                for caption in captions:
                    caption_list.append(caption)
                    video_features.append(features)
            return video_features, caption_list

        #colloate function for dataloader
        def collate_fn(batch, word2idx):
            video_features = []
            tokenized_captions = []

            for video_feature, caption in batch:
                tokens = [word2idx.get(word, word2idx["<UNK>"]) for word in caption.lower().split()]
                tokens = [word2idx["<BOS>"]] + tokens + [word2idx["<EOS>"]]
                tokenized_captions.append(torch.tensor(tokens, dtype=torch.long))

                video_feature = np.load(video_feature)
                video_features.append(torch.tensor(video_feature, dtype = torch.float32))
            padded_captions = pad_sequence(tokenized_captions, batch_first=True, padding_value=word2idx["<PAD>"])
            video_features_tensor = torch.stack(video_features)
            return video_features_tensor, padded_captions

        #dataset class to feed into dataloader
        class VideoCaptionDataset(Dataset):
            def __init__(self, video_features, captions):
                self.video_features = video_features
                self.captions = captions
            def __len__(self):
                return len(self.captions)
            def __getitem__(self, idx):
                return self.video_features[idx], self.captions[idx]

        #optimizer and loss function for training
        optimizer = torch.optim.Adam(seq2seq.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss(ignore_index=word2idx["<PAD>"])

        #load features, match them, and fead it into dataloader
        train_video_features = load_video_features("training_data/feat")
        matched_data = match_labels(train_video_features, "training_label.json")
        video_features_tensor, caption = dataset_prep(matched_data)
        dataset = VideoCaptionDataset(video_features_tensor, caption)
        dataloader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=True,
            collate_fn=lambda batch: collate_fn(batch, word2idx)
        )

        #training
        losses = []
        for epoch in range(200):
            seq2seq.train()
            epoch_loss = 0

            for video_features_batch, captions_batch in dataloader:
                video_features_batch = video_features_batch.to(device)
                captions_batch = captions_batch.to(device)

                optimizer.zero_grad()

                outputs = seq2seq(video_features_batch, captions_batch)
                outputs = outputs.view(-1, output_dim).to(device)
                targets = captions_batch.view(-1).to(device)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / len(dataloader)
            losses.append(avg_epoch_loss)

            print(f'Epoch [{epoch + 1}/{200}]')

        #save the model after training
        torch.save(seq2seq.state_dict(), 'seq2seq_model.pth')

        plt.figure()
        plt.plot(range(1, len(losses) + 1), losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Time')
        plt.legend()
        plt.savefig('training_loss_plot.png')  # Save the plot as a PNG file
        plt.show()

    #if there are two arguments it will test the model
    elif len(sys.argv) == 3:
        # function to load video features for test, differs from train because
        # train copies just the path, given the size of training data
        def load_video_features_test(directory):
            video_features = []
            for filename in os.listdir(directory):
                if filename.endswith(".npy"):
                    video_id = os.path.splitext(filename)[0]
                    features_path = os.path.join(directory, filename)
                    features = torch.tensor(np.load(features_path), dtype=torch.float32).to(device)
                    video_features.append((video_id, features))
            return video_features

        #dataset creator for training,
        class TestVideoDataset(Dataset):
            def __init__(self, video_features):
                self.video_features = video_features
            def __len__(self):
                return len(self.video_features)
            def __getitem__(self, idx):
                video_id, features = self.video_features[idx]
                return video_id, features

        #load model and set to eval mode
        seq2seq.load_state_dict(torch.load('seq2seq_model.pth', map_location=device))
        seq2seq.eval()
        #load video features for test
        test_video_features = load_video_features_test(sys.argv[1])
        test_video_features = TestVideoDataset(test_video_features)
        #initialize dataloader
        dataloader = DataLoader(
            test_video_features,
            batch_size=32,
            shuffle=True
        )

        #variable to hold results
        results = []
        #evaluate against testing data
        with torch.no_grad():
            for batch in dataloader:
                video_id, features_batch = batch
                features_batch = features_batch.to(device)

                encoder_output, hidden, cell = encoder(features_batch)

                batch_size = features_batch.size(0)
                input_word = torch.full((batch_size, 1), word2idx["<BOS>"], dtype=torch.long).to(device)

                generated_captions = [[] for _ in range(batch_size)]

                for t in range(30):
                    prediction, hidden, cell = seq2seq.decoder(input_word, hidden, cell, encoder_output)
                    predicted_word = prediction.argmax(dim=1)
                    input_word = predicted_word

                    for i in range(batch_size):
                        word_idx = predicted_word[i].item()
                        if word_idx == word2idx["<EOS>"]:
                            break
                        generated_captions[i].append(idx2word[word_idx])

                for i in range(batch_size):
                    caption_text = ' '.join(
                        [word for word in generated_captions[i]])
                    vid_id = video_id[i]
                    results.append(f"{vid_id},{caption_text}")

        #write the results into an output file
        with open(sys.argv[2], 'w') as f:
            for item in results:
                f.write(item + "\n")

    else:
        print("Incorrect number of arguments")
        print("Usage: for training model_seq2seq.py")
        print("Usage: for testing model.seq2seq.py <testing data> <output file>")

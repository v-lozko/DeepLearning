Items to note regarding execution of the file:
1. hw2_seq2seq.sh ran on the palmetto servers correctly, though I had to use dos2unix command. I am assuming it will run on local device as well. 
2. the .sh file also runs bleu_eval.py after generation of the output file
3. the model requires access to the training data to generate the vocabulary. specifically, relative to the model_seq2seq.py file this is passed in to generate the vocab: "training_label.json"

please reach out if you have any further questions.

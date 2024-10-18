Items to note regarding execution of the file:
1. hw2_seq2seq.sh ran on the palmetto servers correctly. For it to work on palmetto i did have to add 'module load ngc/pytorch/23.06' and 'module load cuda/12.3.0' as the PATH changed when running the script. I did remove them from the version i uploaded here. I am assuming the version uploaded will run for you as you wont run it on palmetto.
2. the model was too large for github, I added autodownload to the .sh file and it did work for me, i made permissions so file is open to everyone.
3. the .sh file also runs bleu_eval.py after generation of the output file
4. the model requires access to the training data to generate the vocabulary. specifically, relative to the model_seq2seq.py file this is passed in to generate the vocab: "training_label.json"

please reach out if you have any further questions. valeryl@g.clemson.edu

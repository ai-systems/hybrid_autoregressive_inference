"""
Code to reproduce the results on the multi-hop explanation regeneration task ( https://github.com/umanlp/tg2019task )
presented in "Hybrid Autoregressive Inference for Scalable Multi-hop Explanation Regeneration" (AAAI 2022)
"""
import math
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer,  SentencesDataset, LoggingHandler, losses, InputExample, models
from sentence_transformers.losses.ContrastiveLoss import  SiameseDistanceMetric
from sentence_transformers.evaluation import InformationRetrievalEvaluator
import logging
from datetime import datetime
import json
import csv

with open("./data/training/corpus.json", "rb") as f: 
    corpus = json.load(f) 

with open("./data/training/explanations_dev.json", "rb") as f: 
    explanations = json.load(f) 

with open("./data/training/queries_dev.json", "rb") as f: 
    queries = json.load(f) 

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# parameters
train_batch_size = 16
num_epochs = 3
model_save_path = './models/autoregressive_bert_biencoder'

#create a new model e.g. "scibert" 'allenai/scibert_scivocab_uncased'
word_embedding_model = models.Transformer('bert-base-uncased', max_seq_length = 128)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for trainings

logging.info("Read train dataset")

train_examples = []

with open('./data/training/chains_train.csv') as csvfile:
  spamreader = csv.reader(csvfile, delimiter='\t')
  count = 0
  for row in spamreader:
    score = float(row[2])
    if score < 1.0:
      score = 0.0
    train_examples.append(InputExample(texts = [str(row[0]), str(row[1])], label = int(score)))


train_dataset = SentencesDataset(train_examples, model)
train_dataloader = DataLoader(train_dataset, shuffle = True, batch_size = train_batch_size)
train_loss = losses.ContrastiveLoss(model = model, margin = 0.25)


evaluator = InformationRetrievalEvaluator(queries, corpus, explanations, show_progress_bar = True, batch_size = 16, ndcg_at_k= [10000], map_at_k = [10000])

# Configure the training
warmup_steps = math.ceil(len(train_examples)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=200,
          warmup_steps=warmup_steps,
          output_path=model_save_path,
          use_amp=True)
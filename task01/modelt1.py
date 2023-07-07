from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import argparse
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report
import os
import logging as log

# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# Load data
from print_stat import print_information

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="bert-base-cased")
parser.add_argument('--model_type', required=False, help='model type', default="bert")
parser.add_argument('--epochs', required=False, default= 4)
arguments = parser.parse_args()

train_df = pd.read_csv('training.tsv', sep="\t")
val_df = pd.read_csv('validation.tsv', sep="\t")

train_df = train_df[['tweet_id', 'text', 'labels']]
# ids_to_select = val_df['id'].astype(int)
# selected_labels = train_df[train_df['id'].isin(ids_to_select)]['labels'].astype(int)
# merged_df = train_df.merge(val_df, on='id', how='inner')
# selected_labels = merged_df['labels']
# # matched_df = merged_df['id']
# matched_ids = val_df[val_df['id'].isin(train_df['id'])]['id'].tolist()
# new_df = pd.DataFrame({'id': matched_ids, 'labels': selected_labels})
# print(new_df)
#
# new_df.to_csv('test.tsv', sep='\t', index=False)

test_sentences = val_df['text'].tolist()

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
EPOCHS = int(arguments.epochs)

# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": EPOCHS,
             "train_batch_size": 8,
             "use_multiprocessing": False,
             "use_multiprocessing_for_evaluation":False,
             "n_fold":1,
        }


# Create a ClassificationModel
model = ClassificationModel(
    MODEL_TYPE, MODEL_NAME,
    args=train_args, use_cuda=torch.cuda.is_available()
)

# Train the model
model.train_model(train_df)

# Make predictions with the model
predictions, raw_outputs = model.predict(test_sentences)
print(predictions)

val_df['prediction'] = predictions
print(val_df)

# print_information(val_df, "pred", "labels")
dpred = val_df[['tweet_id', 'prediction']].copy()
dpred.to_csv('valid2.tsv', sep='\t', index=False)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(val_df)

dtruth = pd.read_csv('test.tsv', sep='\t')
dtruth.set_index('tweet_id', inplace=True)


dpred = pd.read_csv('valid2.tsv', sep='\t')

dpred.set_index('tweet_id', inplace=True)
dpred.rename(columns={"prediction": "pred"}, inplace=True)

assert len(dtruth) == len(dpred)

dEval = pd.concat([dtruth, dpred], axis=1, join='inner')
prec = precision_score(list(dEval['labels']), list(dEval['pred']), pos_label=1, average='binary')
rec = recall_score(list(dEval['labels']), list(dEval['pred']), pos_label=1, average='binary')
f1 = f1_score(list(dEval['labels']), list(dEval['pred']), pos_label=1, average='binary')

#print(f"cf matrix: {cf}\ncr :{cr}\nPrec:{prec}, Rec:{rec}, F1:{f1}")

log.warning("scores computed")
print("scores computed")

# the scores for the leaderboard must be in a file named "scores.txt"
# https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
with open(os.path.join('scores.txt'), 'w') as output_file:
    output_file.write("Task4F: " + str(f1)+"\n")
    output_file.write("Task4P: " + str(prec)+"\n")
    output_file.write("Task4R: " + str(rec)+"\n")
    output_file.flush()

log.warning("output file written")
print("output file written")

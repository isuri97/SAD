from simpletransformers.classification import ClassificationModel, ClassificationArgs
import pandas as pd
import argparse
import torch

# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# Load data

parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
arguments = parser.parse_args()

train_df = pd.read_csv('training.tsv', sep="\t")
val_df = pd.read_csv('validation.tsv', sep="\t")

# train_df = train_df[['tweet_id', 'text', 'label']]
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

# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": 4,
             "train_batch_size": 8,
             "use_multiprocessing": False,
             "use_multiprocessing_for_evaluation":False,
             "n_fold":1,
        }

MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
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

val_df['label'] = predictions
print(val_df)

new_df2 = val_df[['id', 'label']].copy()
new_df2.to_csv('valid2.tsv', sep='\t', index=False)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(val_df)




from simpletransformers.classification import ClassificationModel
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse


# logging.basicConfig(level=logging.INFO)
# transformers_logger = logging.getLogger("transformers")
# transformers_logger.setLevel(logging.WARNING)

# Load data
parser = argparse.ArgumentParser(
    description='''evaluates multiple models  ''')
parser.add_argument('--model_name', required=False, help='model name', default="xlm-roberta-large")
parser.add_argument('--model_type', required=False, help='model type', default="xlmroberta")
parser.add_argument('--epochs', required=False, default= 4)
arguments = parser.parse_args()

train_df = pd.read_csv('train-dataset.csv', sep=",")
val_df = pd.read_csv('val-dataset.csv', sep=",")
# combined_df = pd.read_csv('reduced-combined.csv', sep=",")

train_df = train_df[['text', 'labels']]
# combined_df = combined_df[['text','labels']]
#
# train_df = pd.concat([train_df, combined_df], ignore_index=True)
# train_df = train_df.sample(frac=1).reset_index(drop=True)

# train_df = train_df[['id', 'text', 'labels']]
# # train_df = train_df[['text', 'labels']]
# combined_df = combined_df [['text','labels']]





# ids_to_select = val_df['id'].astype(int)
# selected_labels = train_df[train_df['id'].isin(ids_to_select)]['labels'].astype(int)
# merged_df = train_df.merge(val_df, on='id', how='inner')
# selected_labels = merged_df['labels']
# # matched_df = merged_df['id']
# matched_ids = val_df[val_df['id'].isin(train_df['id'])]['id'].tolist()
# new_df = pd.DataFrame({'id': matched_ids, 'labels': selected_labels})
# print(new_df)

# new_df.to_csv('test.tsv', sep='\t', index=False)

# Optional model configuration
# train_df = train_df + combined_df
train_set =train_df
# train_set, validation_set = train_test_split(train_df, test_size=0.1)



# print(validation_set)
test_sentences = val_df['text'].tolist()

# model_args.wandb_project="holo-ner"
MODEL_NAME = arguments.model_name
MODEL_TYPE = arguments.model_type
EPOCHS = int(arguments.epochs)

# define hyperparameter
train_args = {"reprocess_input_data": True,
             "overwrite_output_dir": True,
             "fp16":False,
             "num_train_epochs": EPOCHS,
             "train_batch_size": 4,
             "use_multiprocessing": False,
             "use_multiprocessing_for_evaluation":False,
             "n_fold":1,
             "wandb_project":"SAD",
             # "use_early_stopping ":True,
              # "evaluate_during_training":True,
              # "save_best_model":True,
              "learning_rate": 4e-5
        }

# # Create a ClassificationModel
# model = ClassificationModel(
#     MODEL_TYPE, MODEL_NAME,
#     args=train_args
# )
#
# # Train the model
# model.train_model(combined_train, eval_df=combined_val)
#
#
# train_args['num_train_epochs']=EPOCHS


#further finetune on task dataset

# Create a ClassificationModel
model = ClassificationModel(
    MODEL_TYPE, MODEL_NAME,
    args=train_args
)

# Train the model
# model.train_model(train_set, eval_df=validation_set)
model.train_model(train_set)

# Make predictions with the model
predictions, raw_outputs = model.predict(test_sentences)
print(predictions)

val_df['label'] = predictions
print(val_df)

new_df2 = val_df[['id', 'label']].copy()
new_df2.to_csv('valid2.tsv', sep='\t', index=False)

# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(validation_set)




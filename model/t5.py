import pandas as pd
from simpletransformers.t5 import T5Model, T5Args
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('modified-train.csv', sep=",")
val_df = pd.read_csv('val-dataset-copy.csv', sep=",")

# train_df= train_df.columns = ["text", "labels"]
# train_df['text'] = train_df['text'].astype(str)

train_df = train_df[['prefix','input_text','target_text']]
train_df['target_text'] = train_df['target_text'].astype(str)

# train_set, validation_set = train_test_split(train_df, test_size=0.1)

# eval_df = pd.DataFrame(eval_data)
# eval_df.columns = ["prefix", "input_text", "target_text"]
# eval_df['target_text'] = eval_df['target_text'].astype(str)
# text_add = 'binary classification:'
val_df['text'] = 'binary classification:' + val_df['text'].astype(str)
test_sentences = val_df['text'].tolist()

# validation_set
# validation_set.columns = ["prefix", "input_text", "target_text"]
# validation_set['target_text'] = validation_set['target_text'].astype(str)


# Configure the model
model_args = T5Args()
model_args.num_train_epochs = 10
model_args.no_save = True
model_args.evaluate_generated_text = False
model_args.evaluate_during_training = False
model_args.evaluate_during_training_verbose = True
model_args.max_length = 512
model_args.use_multiprocessing= False
model_args.use_multiprocessing_for_evaluation=False
model_args.evaluate_each_epoch = False

model = T5Model("t5", "t5-base", args=model_args)

# Train the model
model.train_model(train_df)

# Evaluate the model
# result = model.eval_model(validation_set)

predictions = model.predict(test_sentences)
print(predictions)

val_df['label'] = predictions
print(val_df)

new_df2 = val_df[['id', 'label']].copy()
new_df2.to_csv('valid2.tsv', sep='\t', index=False)
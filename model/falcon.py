from langchain import HuggingFacePipeline
from langchain import PromptTemplate,  LLMChain
from transformers import AutoTokenizer, pipeline,AutoModelForCausalLM
import pandas as pd
import transformers

import torch

# from util.print_stat import print_information

# set env
# os.environ('')


model = "tiiuae/falcon-7b-instruct" #tiiuae/falcon-40b-instruct

tokenizer = AutoTokenizer.from_pretrained(model)

pipeline = pipeline(
    "text-generation", #task
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="auto",
    max_length=500,
    do_sample=True,
    top_k=1,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id
)

# llm = AutoModelForCausalLM.from_pretrained(pipeline=pipeline, model_kwargs={'temperature':0})
llm = HuggingFacePipeline(pipeline=pipeline, model_kwargs={'temperature':0})
template = """Posts containing any form of conveys self reporting of social anxiety, diagnosed with SAD, doctor agrees or believes on SAD, attended to therapest for SAD,
taking medicine for SAD, which can be veiled or direct are social anxiety posts. 
This includes posts of self reported the social anxiety (SA). Posts that do not contain any information related to social Anxiety, not diagnosed as social anxiety, having
diagnosed with other disease, just going to the counselor, just self diagnosed without doctor and taking medicine without prescription marked as zero.
Question: {question}
Answer:"""
prompt = PromptTemplate(template=template, input_variables=["question"])

llm_chain = LLMChain(prompt=prompt, llm=llm)

test = pd.read_csv('test_set.csv', sep=",")
test = test[['id', 'text']]
final_predictions = []

for index, row in test.iterrows():
    question = "Can this post accurately as self reporting  social anxiety disorder? Comment: " + row['text']
    response = llm_chain.run(question)
    if response.split(',')[0].strip() == "Yes":
        final_predictions.append(int(1))
    else:
        final_predictions.append(int(0))


test['label'] = final_predictions
# print_information(test, "predictions", "Class")

print(test)

new_df2 = test[['id', 'label']].copy()
new_df2.to_csv('valid2.tsv', sep='\t', index=False)

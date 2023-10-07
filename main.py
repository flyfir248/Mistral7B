from flask import Flask, request, jsonify
import textwrap
import json
import torch
import transformers
from transformers import GenerationConfig, pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

app = Flask(__name__)

tokenizer = AutoTokenizer.from_pretrained("ehartford/samantha-mistral-7b")

model = AutoModelForCausalLM.from_pretrained("ehartford/samantha-mistral-7b")


#model = AutoModelForCausalLM.from_pretrained("ehartford/samantha-mistral-7b",
#                                              load_in_8bit=True,
#                                              device_map='auto',
#                                              torch_dtype=torch.float16,
#                                              low_cpu_mem_usage=True,
#                                              )

tokenizer.eos_token_id, tokenizer.pad_token_id
tokenizer.pad_token_id = 0
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=1536,
    temperature=0.7,
    top_p=0.95,
    repetition_penalty=1.15
)

addon_prompt = "Your name is Samantha."
system_prompt = "A chat between a curious user and an artificial intelligence assistant. \nThe assistant gives helpful, detailed, and polite answers to the user's questions."

def get_prompt(human_prompt):
    prompt_template = f"{addon_prompt}\n{system_prompt}\n\nUSER: {human_prompt} \nASSISTANT: "
    return prompt_template

def remove_human_text(text):
    return text.split('USER:', 1)[0]

def parse_text(data):
    for item in data:
        text = item['generated_text']
        assistant_text_index = text.find('ASSISTANT:')
        if assistant_text_index != -1:
            assistant_text = text[assistant_text_index+len('ASSISTANT:'):].strip()
            assistant_text = remove_human_text(assistant_text)
            wrapped_text = textwrap.fill(assistant_text, width=100)
            return wrapped_text

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json['user_query']
    prompt = get_prompt(user_query)
    raw_output = pipe(prompt)
    response = parse_text(raw_output)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)

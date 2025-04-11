#from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList
#import torch

from openai import OpenAI
import json



def _handle_caii_request(prompt: str):
  # If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
  API_KEY = json.load(open("/tmp/jwt"))["access_token"]
  MODEL_ID = "meta/llama-3.1-8b-instruct"

  client = OpenAI(
    base_url="https://ml-2dad9e26-62f.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/goes---llama3-8b-throughput/v1",
    api_key=API_KEY,
  )

  completion = client.chat.completions.create(
    model=MODEL_ID,
    messages=[{"role": "user", "content": prompt}],
    temperature=0.2,
    top_p=0.7,
    max_tokens=1024,
    stream=False
  )

  response_text = completion.choices[0].message.content
  return response_text

def get_llm_generation(prompt, stop_words, temperature=0.7, max_new_tokens=256, top_p=0.85, top_k=70, repetition_penalty=1.07, do_sample=False):
  return _handle_caii_request(prompt)

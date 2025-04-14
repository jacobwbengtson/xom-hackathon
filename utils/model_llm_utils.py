from openai import OpenAI
import json
import os



def _handle_caii_request(prompt: str):
  # If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
  API_KEY = json.load(open("/tmp/jwt"))["access_token"]
  MODEL_ID = os.environ.get("LLM_MODEL_ID")

  client = OpenAI(
    base_url=os.environ.get("LLM_MODEL_ENDPOINT"),
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

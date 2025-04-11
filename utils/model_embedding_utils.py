


from openai import OpenAI
import json
import tiktoken


def _handle_caii_request(sentence: str):
  # If running from workbench use /tmp/jwt. Otherwise provide your CDP_TOKEN
  API_KEY = json.load(open("/tmp/jwt"))["access_token"]

  # Set "query" in MODEL_ID for user questions to find matches in the vector store. Use "passage" when embedding reference data into the vector store.
  MODEL_ID = "nvidia/nv-embedqa-e5-v5-passage"

  client = OpenAI(
    base_url="https://ml-2dad9e26-62f.go01-dem.ylcu-atmi.cloudera.site/namespaces/serving-default/endpoints/goes---e5-embeddings/v1",
    api_key=API_KEY,
  )

  embedding = client.embeddings.create(
    input=sentence, 
    model=MODEL_ID
  ).data[0].embedding
  return embedding


# This limit is imposed by the embdedding model in use in cai (e5 embdeddings, 512 tokens) Setting this to a value well below that to avoid the limit.
EMBEDDING_CTX_LENGTH = 450
EMBEDDING_ENCODING = 'cl100k_base'

def truncate_text_tokens(text, encoding_name=EMBEDDING_ENCODING, max_tokens=EMBEDDING_CTX_LENGTH):
    """Truncate a string to have `max_tokens` according to the given encoding."""
    encoding = tiktoken.get_encoding(encoding_name)
    truncated_tokens =  encoding.encode(text)[:max_tokens]
    return encoding.decode(truncated_tokens)


def get_embeddings(sentence):
    
    return _handle_caii_request(truncate_text_tokens(sentence))
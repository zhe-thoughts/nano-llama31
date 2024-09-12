"""
Takes any pure text file and converts it to a binary file using 
the llama3_1 tokenizer.
"""

import os
import numpy as np
import sys
import random
from tokenizer import Tokenizer
import ray

ray.init()

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
tokenizer_path = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model"

ds = ray.data.read_text(input_file_path)

tokenizer = Tokenizer(tokenizer_path)
def encode_batch(batch):
    return [{'token': tokenizer.encode(x['text'], bos=True, eos=True)} for x in batch]

# Use map_batches to apply the encode function to batches of data
tokens_ds = ds.map_batches(encode_batch, batch_size=1000)  # Adjust batch_size as needed

tokens = [x['token'] for x in tokens_ds.take_all()]
tokens = [item for sublist in tokens for item in sublist]

assert len(tokens) < 2**31, "token count too large" # ~2.1B tokens

 # construct the header
header = np.zeros(256, dtype=np.int32)
header[0] = 20240801 # magic
header[1] = 7 # version
header[2] = len(tokens) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
# construct the tokens numpy array, if not already
tokens_np = np.array(tokens, dtype=np.uint32)
    
# write to file
print(f"writing {len(tokens):,} tokens to {output_file_path}")
with open(output_file_path, "wb") as f:
    f.write(header.tobytes())
    f.write(tokens_np.tobytes())

ray.shutdown()
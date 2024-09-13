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
num_cpus = 200

input_file_path = sys.argv[1]
output_file_path = sys.argv[2]
tokenizer_path = "llama-models/models/llama3_1/Meta-Llama-3.1-8B/tokenizer.model"

# Read the input file
with open(input_file_path, 'r') as f:
    data = f.readlines()

print(f"finished reading: {len(data):,} lines")

# Initialize the tokenizer
tokenizer = Tokenizer(tokenizer_path)

@ray.remote
def process_batch(batch):
    tokens = []
    for line in batch:
        line = line.strip()
        if not line:
            continue
        tokens += tokenizer.encode(line, bos=True, eos=True)

    assert len(tokens) < 2**31, "token count too large" # ~2.1B tokens

    # construct the header
    header = np.zeros(256, dtype=np.int32)
    header[0] = 20240801 # magic
    header[1] = 7 # version
    header[2] = len(tokens) # number of tokens after the 256*4 bytes of header (each 2 bytes as uint16)
    # construct the tokens numpy array, if not already
    tokens_np = np.array(tokens, dtype=np.uint32)

    random_number = random.randint(1000, 9999)
    output_file_path = f"{output_file_path}_{random_number}.bin"
        
    # write to file
    print(f"writing {len(tokens):,} tokens to {output_file_path}")
    with open(output_file_path, "wb") as f:
        f.write(header.tobytes())
        f.write(tokens_np.tobytes())

# Split data into batches
batch_size = int(len(data) / num_cpus)  # Adjust batch_size as needed
batches = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]

# Distribute the batches to the remote function
futures = [process_batch.remote(batch) for batch in batches]
print(f"launched {len(batches):,} batches")

# Collect the results
results = ray.get(futures)

ray.shutdown()
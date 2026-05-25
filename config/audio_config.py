'''
Why 16000?
    Because:
        16kHz sample rate
    means:
        16000 samples/sec
    So chunk size = 16000 => 1 second audio chunk

    2 second chunk size = 32000
    1/2 second chunk size = 8000
Why chunking

Because transformers and DACs train on:
fixed-length segments
NOT arbitrary audio lengths.
Example
Raw audio: 13.7 seconds
At 16kHz: 13.7×16000=219200 samples.
Too large/unpredictable.
So chunk into: 1 sec windows
Result: 13 chunks
Now training batches become easy.
| Chunk Size | Meaning | Good For            |
| ---------- | ------- | ------------------- |
| 8000       | 0.5 sec | tiny experiments    |
| 16000      | 1 sec   | BEST starting point |
| 32000      | 2 sec   | better context      |
| 64000      | 4 sec   | larger GPUs         |

'''
SAMPLE_RATE = 16000

CHUNK_SIZE = 16000
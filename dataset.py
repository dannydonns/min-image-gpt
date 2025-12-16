# will contain general classes for the dataset
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import pickle


# this is the file where I keep all of my tokenizers
class BPETokenizer:
    # this takes in a large corpus of text and tokenizes it
    def __init__(self, vocab_size: int = 1000, max_seq: int = 512):
        self.max_seq = max_seq
        self.vocab_size = vocab_size
        
        # Base vocabulary: Map 0-255 to their byte values
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        
        # Merges: Map (id1, id2) -> new_id
        self.merges = {}

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_ids(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            # If we are not at the last element and the pair matches
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def bpe(self, text):
        # 1. Convert the entire text to UTF-8 integers (0-255)
        ids = list(text.encode('utf-8'))

        # Calculate how many merges we need to perform to hit target vocab_size
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            
            # If no pairs exist (text is too short), stop
            if not stats:
                break

            # 2. Find the most frequent pair
            pair = max(stats, key=stats.get)
            
            # Create a new token ID
            idx = 256 + i
            
            # 3. Record the merge rule
            self.merges[pair] = idx
            
            # Update vocab mapping (reconstruct bytes from the constituent parts)
            # The byte representation of the new token is the concatenation of its parts
            self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]
            
            # 4. Apply the merge to the current text to prep for next iteration
            ids = self._merge_ids(ids, pair, idx)
            
            # Optional: Print progress
            # print(f"Merging {pair} into new token {idx}")

    def tokenize(self, text):
        # Start with raw UTF-8 bytes
        ids = list(text.encode('utf-8'))
        
        while len(ids) >= 2:
            # Find all possible pairs in the current sequence
            stats = self._get_stats(ids)
            
            # We only care about pairs that we have learned in self.merges
            # We want to find the pair with the LOWEST new_id (applied earliest during training)
            # This reconstructs the merge order.
            pair_to_merge = None
            min_merge_idx = float('inf')

            for pair in stats:
                if pair in self.merges:
                    if self.merges[pair] < min_merge_idx:
                        min_merge_idx = self.merges[pair]
                        pair_to_merge = pair
            
            # If no mergeable pairs are found, we are done
            if pair_to_merge is None:
                break
            
            # Apply the merge
            ids = self._merge_ids(ids, pair_to_merge, min_merge_idx)
            
        return ids

    def detokenize(self, tokens):
        # 1. Retrieve bytes for each token from the vocab
        byte_stream = b"".join(self.vocab[idx] for idx in tokens)
        
        # 2. Decode bytes to string (errors='replace' handles invalid utf-8 sequences gracefully)
        text = byte_stream.decode('utf-8', errors='replace')
        return text
    
    def add_specials(self):
        # these are the usual special tokens that are in a 
        specials = [
            "[CLS]",
            "[SEP]",
            "[UNK]",
            "[PAD]",
            "[BOS]",
            "[EOS]"
        ]

        # check if we have already done this
        specials_encoded = []
        for spec in specials:
            spec_encoded = spec.encode('utf-8')
            if spec_encoded in self.vocab:
                return 1
            specials_encoded.append(spec_encoded)
        
        
        start = len(self.vocab)
        for spec in specials:
            spec_encoded = spec.encode('utf-8')
            self.vocab[spec_encoded] = start
            start += 1
        return 0
    
    def save(self, filename):
        """
        Saves the tokenizer state (vocab, merges, config) to a file.
        """
        # We wrap the state in a simple dictionary
        state = {
            "vocab_size": self.vocab_size,
            "max_seq": self.max_seq,
            "vocab": self.vocab,
            "merges": self.merges
        }
        
        # 'wb' means Write Binary - required for pickle
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
        print(f"Tokenizer saved to {filename}")

    @classmethod
    def load(cls, filename):
        """
        Loads a tokenizer from a file.
        Usage: tokenizer = BPETokenizer.load("my_tokenizer.model")
        """
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        
        # Create a new instance of the class
        # We use cls() instead of BPETokenizer() to support inheritance
        tokenizer = cls(vocab_size=state["vocab_size"], max_seq=state["max_seq"])
        
        # Restore the internal state
        tokenizer.vocab = state["vocab"]
        tokenizer.merges = state["merges"]
        
        print(f"Tokenizer loaded from {filename}")
        return tokenizer


class ShakespeareDataset(Dataset):
    # dataset for predicting the next word/token in text
    # 
    # Example:
    #   input:  To be or not to
    #   output: be or not to be
    #
    # This is not an *exact* example of what will happen,
    # since tokens are not based on the words themselves.
    # But I think this is the gist of what I should be
    # doing here. 

    def __init__(self, block_size: int):
        super().__init__()
        
        # initial parameters
        self.block_size = block_size

        # tokenizer
        self.tokenizer = BPETokenizer.load("shakespeare_tokenizer.pkl")

        # load in text
        self.data = self._load_text()

    def _load_text(self):
        # loads in text from the shakespeare file and tokenizes it
        # load text

        file_path = 'shakespeare.txt'
        try:
            with open(file_path, 'r') as file:
                text_corpus = file.read()
            print("File content loaded successfully:")
            print(text_corpus)
        except FileNotFoundError:
            print(f"Error: The file '{file_path}' was not found.")
        except Exception as e:
            print(f"An error occurred: {e}")

        print("Loading Shakespeare Dataset...")
        # tokenize  
        return self.tokenizer.tokenize(text_corpus)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]
        y = chunk[1:]
        return x, y
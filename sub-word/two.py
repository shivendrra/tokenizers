import os
current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)

text = """
- Hmm. Two clear indicators that
your recruiter is a criminal. I'm sorry, baby bro. I know
how much you love pianos. - I already gave the guy the $500. I sent him a payment digitally. (footsteps knock) - Here, Jordan, why
don't you come with me? I'll show you how to
report the scam to the FTC. Then we'll go over whichever
bank or payment platform you used also. (Loretta sighs) (tense music) - Transfer receipts. This guy's a professional
and he isn't operating alone. Travel in 88. This isn't over. - With so many folks doing
their banking online, it's easier than ever for criminals to try and impersonate banks. In fact, this show was inspired by Steven's encounter with a scammer. - It's true. But instead
of bringing me down, it fueled my passion. - You know, Steven came to me and he said, "Betsy, I know you retired early and moved to an RV in the Salt Flats, but I've got a thrilling idea for a show." And here we are. (both laugh) - You're my twin? - Yeah. (both laugh) - Let's check out how the squad deals with a banking scam. (calm music) - Romance, man. What a bunch of baloney. - Ah, I think it's sweet. - There's nothing sweet
about tender moments and grand gestures, Skip. It's like get your own self-esteem. - Oh, come on, Ace. You telling me that
you've never been in love? - My savings account compromised? No, I didn't authorize
a $12,000 withdrawal. That's my life savings. Of course you're speaking to the real me. My social security number. It's 131. Hey! - Why don't you come with us? We'll explain on the way. (footsteps knock) - Is this the guy? - I've been saving that
money for years, man. I was gonna take my girlfriend to Palermo and hide an engagement ring
inside an arancini ball. (paper rustles) - Palermo is beautiful this time of year. We won't let that dream die. - All right, first thing, Benji, we gotta make sure that your account is actually compromised. Like I know it's only my
second day on the job, but this feels like some funny business. - My life is over. - Benji, focus. Call the number on the
back of your debit card. That's a secure way to see if your account has been compromised. - I called the number. They said my account is secure after all. - You know Benji, a
bank will never call you to ask for personal
information, or even an OTP. - OTP? - One-time passcode. The next time it happens, you'll know it's a criminal on the phone. - I really appreciate it. You said Palermo was
beautiful this time of year. You been? - You said you were going to propose. (Loretta scoffs) Take it from me, kid. Don't wait. - [Betsy] You all right? - Yeah. Next, you are going to
see the squad dealing with their toughest case
yet, a government scam. - Now, I know a lot of things famously, but I had no idea how prevalent
these kinds of scammers are. Some of the most common
government scams include Covid scams, social security scams, and as you're about to see, IRS scams.
"""

import regex as re
from collections import Counter

class BPETokenizer:
    def __init__(self, vocab_size=1000, special_tokens=["[UNK]", "[PAD]", "[CLS]", "[SEP]"]):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens
        self.regex_pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+ | ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def _build_vocab(self, text):
      token_counts = Counter(text)
      vocab = {token: count for token, count in token_counts.items() if token in self.special_tokens}
      if not vocab:
          return self.special_tokens
      while len(vocab) < self.vocab_size:
          most_common_pair = max(self._get_token_pairs(vocab), key=vocab.get)
          if most_common_pair not in vocab:
              break
          new_token = "".join(most_common_pair)
          vocab[new_token] = vocab[most_common_pair[0]] + vocab[most_common_pair[1]]
          del vocab[most_common_pair[0]]
          del vocab[most_common_pair[1]]
      return vocab

    def _get_token_pairs(self, vocab):
        token_pairs = Counter()
        for token, count in vocab.items():
            token_chars = token.split()
            for i in range(len(token_chars) - 1):
                token_pairs[token_chars[i], token_chars[i + 1]] += count
        return token_pairs

    def fit(self, text):
        tokenized_text = self.regex_pattern.findall(text.lower())
        self.vocab = self._build_vocab(tokenized_text)

    def tokenize(self, text):
        tokenized_text = self.regex_pattern.findall(text.lower())
        tokens = []
        for token in tokenized_text:
            if token in self.vocab:
                tokens.append(token)
            else:
                tokens.extend(self._encode_token(token))
        return tokens

    def _encode_token(self, token):
        if len(token) == 1:
            return [token]
        encoded_tokens = []
        char_buffer = ""
        for char in token:
            char_buffer += char
            if char_buffer in self.vocab or len(char_buffer) == 1:
                encoded_tokens.append(char_buffer)
                char_buffer = ""
        return encoded_tokens

    def decode(self, tokens):
        text = "".join(tokens)
        return text

with open('../captions.txt', 'r', encoding='utf-8') as file:
  train_data = file.read()

bpe_tokenizer = BPETokenizer(vocab_size=1000)
bpe_tokenizer.fit(train_data)
encoded_tokens = bpe_tokenizer.tokenize(text)
decoded_text = bpe_tokenizer.decode(encoded_tokens)

print("Encoded Tokens:", encoded_tokens)
print("Decoded Text:", decoded_text)
print(decoded_text==text)
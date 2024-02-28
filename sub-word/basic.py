text = """
- Hmm. Two clear indicators that your recruiter is a criminal. I'm sorry, baby bro. I know how much you love pianos. - I already gave the guy the $500. I sent him a payment digitally. (footsteps knock) - Here, Jordan, why don't you come with me?
I'll show you how to report the scam to the FTC. Then we'll go over whichever bank or payment platform you used also. (Loretta sighs) (tense music) - Transfer receipts. This guy's a professional and he isn't operating alone. Travel in 88. This isn't over.
- With so many folks doing their banking online, it's easier than ever for criminals to try and impersonate banks. In fact, this show was inspired by Steven's encounter with a scammer. - It's true. But instead
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
import os

current_directory = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_directory)

r"""
  ## space is merged with each word, before it as a prefix
  ## a litlle smaller than pattern2
  regex_pattern1: '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
  
  ## space is added as a preffix to each word, retains all the initial words
  ## smaller than pattern3
  regex_pattern2: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

  ## space is considered a separate token, all words remain original, no loss of words
  ## largest in length
  regex_pattern3: 's|'t|'re|'ve|'m|'ll|'d|[\w']+|[^\s\w\d]+|\s+(?!\S)|\s+
  
  ## spaces are added as a prefix to the words, but some words are missing hence doesn't retains original text
  ## smallest in length, due to some lost words
  regex_pattern4: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+ | ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
"""

regex_pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")


def get_stats(ids):
  counts = {}
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge(ids, pair, idx):
  new_ids = []
  i = 0
  while i < len(ids):
    if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
      new_ids.append(idx)
      i += 2
    else:
      new_ids.append(ids[i])
      i += 1
  return new_ids

def apply_regex(text, pattern):
  text = re.findall(pattern, text)
  return text

def encode(text, regex_pattern):
  outputs = apply_regex(text, regex_pattern)
  tokens = []
  for word in outputs:
    token = list(word.encode('utf-8'))
    tokens.extend(token)
  
  while True:
    stats = get_stats(tokens)
    pair = max(stats, key=lambda p: merges.get(p, float('inf')))
    if pair not in merges:
      break
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens

def decode(ids):
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode('utf-8', errors='replace')
  return text

# with open('../captions.txt', 'r', encoding='utf-8') as file:
#   new_text = file.read()

merges = {}
vocab_size = 656
n_merges = vocab_size - 256
token_list = apply_regex(text, regex_pattern)
ids = []

for words in token_list:
  new_token = list(words.encode('utf-8'))
  ids.extend(new_token)

for i in range(n_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  # print(f"mergeing {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx

vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
  vocab[idx] = vocab[p0] + vocab[p1]

# print(''.join(encode(text, regex_pattern)))
# print(encode(text, regex_pattern))

encoded_tokens = encode(text, regex_pattern)
# print(encoded_tokens)

# print("tokens length", len(token_list))
# print("ids lentgh", len(ids))
# print(f"compression ratio: {len(token_list)/len(ids):.2f}x")
# print("len tokens: ", len(encode(text, regex_pattern)))
# print("len text: ", len(decode(encode(text, regex_pattern))))
# print(decode(encode(text, regex_pattern)) == text) 
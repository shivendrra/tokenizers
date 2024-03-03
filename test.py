"""
test-file to test tokenizers
"""

import os
current_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(current_dir)

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

from miniBPE import RegexTokenizer, GPT4tokenizer, save_model
import timeit

llama_text = """
<|endoftext|>The llama (/ˈlɑːmə/; Spanish pronunciation: [ˈʎama] or [ˈʝama]) (Lama glama) is a domesticated South American camelid, widely used as a meat and pack animal by Andean cultures since the pre-Columbian era.
Llamas are social animals and live with others as a herd. Their wool is soft and contains only a small amount of lanolin.[2] Llamas can learn simple tasks after a few repetitions. When using a pack, they can carry about 25 to 30% of their body weight for 8 to 13 km (5–8 miles).[3] The name llama (in the past also spelled "lama" or "glama") was adopted by European settlers from native Peruvians.[4]
The ancestors of llamas are thought to have originated from the Great Plains of North America about 40 million years ago, and subsequently migrated to South America about three million years ago during the Great American Interchange. By the end of the last ice age (10,000–12,000 years ago), camelids were extinct in North America.[3] As of 2007, there were over seven million llamas and alpacas in South America and over 158,000 llamas and 100,000 alpacas, descended from progenitors imported late in the 20th century, in the United States and Canada.[5]
<|fim_prefix|>In Aymara mythology, llamas are important beings. The Heavenly Llama is said to drink water from the ocean and urinates as it rains.[6] According to Aymara eschatology,<|fim_suffix|> where they come from at the end of time.[6]<|fim_middle|> llamas will return to the water springs and ponds<|endofprompt|>
""".strip()

specials_string = """
<|endoftext|>Hello world this is one document
<|endoftext|>And this is another document
<|endoftext|><|fim_prefix|>And this one has<|fim_suffix|> tokens.<|fim_middle|> FIM
<|endoftext|>Last document!!! 👋<|endofprompt|>
""".strip()

special_tokens = {
  '<|endoftext|>': 100257,
  '<|fim_prefix|>': 100258,
  '<|fim_middle|>': 100259,
  '<|fim_suffix|>': 100260,
  '<|endofprompt|>': 100276
}
os.makedirs('../models', exist_ok=True)
time01 = timeit.default_timer()

name = '../models/regex'
tokenizer = RegexTokenizer()
tokenizer.train(text, 512, verbose=True)
save_model(name)

time02 = timeit.default_timer()

test_str = "Hello, what's your name?"
print(f'total time taken: {(time02 - time01)} secs')
print(tokenizer.encode(test_str))
print(tokenizer.decode(tokenizer.encode(test_str)))
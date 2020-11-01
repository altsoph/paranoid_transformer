# Paranoid Transformer

## TLDR

After all, this project turns to a published neural network generated book. [Check the story behind it in my Medium post](https://medium.com/altsoph/paranoid-transformer-80a960ddc90a).

## Overview


This is an attempt to make an unsupervised text generator with some specific style and form characteristics of text.
Originaly it was published as an entry for [NaNoGenMo 2019](https://github.com/NaNoGenMo/2019/issues/142) (_National Novel Generation Month_ contest).

The general idea behind the _Paranoid Transformer_ project is to build a paranoiac-critical system based on two neural networks.
The first network (_Paranoiac-intrusive Generator_) is a GPT-based tuned conditional language model and the second one (_Critic subsystem_) uses a BERT-based classifier that works as a filtering subsystem, so it selects the best ones from the flow of text passages. Finally, I used an existing handwriting synthesis neural network implementation to generate a nervous handwritten diary where a degree of shakiness depends on the sentiment strength of a given sentence.

## Generator subsystem

The first network, Paranoiac-intrusive subsystem AKA Generator, uses an [OpenAI GPT](https://github.com/openai/finetune-transformer-lm) architecture and the [implementation from huggingface](https://github.com/huggingface/transformers). I took a publicly available network model already pre-trained on a huge fiction [BooksCorpus dataset](https://arxiv.org/pdf/1506.06724.pdf) with approx ~10K books and ~1B words.

Next, I finetuned it on several additional handcrafted text corpora (altogether ~50Mb of text):
-  a collection of Crypto Texts (Crypto Anarchist Manifesto, Cyphernomicon, etc),
-  another collection of fiction books (from such cyberpunk authors as Dick, Gibson, and others + non-cyberpunk authors, for example, Kafka and Rumi),
-  transcripts and subtitles from some cyberpunk movies and series,
-  several thousands of quotes and fortune cookie messages collected from different sources.

During the fine-tuning phase, I used special labels for conditional training of the model:
  - _QUOTE_ for any short quote or fortune, _LONG_ for others
  - _CYBER_ for cyber-themed text and _OTHER_ for others.
Each text got 2 labels, for example, it was _LONG_+_CYBER_ for Cyphernomicon, _LONG_+_OTHER_ for Kafka and _QUOTE_+_OTHER_ for fortune cookie messages. Note, there were almost no texts labeled as _QUOTE_+_CYBER_, just a few nerd jokes.

At last, in generation mode, I kindly asked the model to generate only _QUOTE_+_CYBER_ texts.
The raw results were already promising enough:

> terosexuality is pleasures a turn off ; and to me not to be a true blossoming beautiful being is on the other side. the wind is our song, the emotions are our wind and a piano, new things change, new smells kick off in time, a spiritually shifting dust. let your eyes sing music for a while. let your ears measure the bass beat of your soul, the gentle winding of the song. then your ears achieve harmony. you can listen to french playstation on live music together forever, in the philly coffeehouse, in them congressional district of the franklin gap building. let painting melt away every other shred of reason and pain, just lew the paint to move thoughts away from blizzes in death. let it dry out, and turn to cosmic delights, to laugh on the big charms and saxophones and fudatron steames of the sales titanium. we are god's friends, the golden hands on the shoulders of our fears. do you knock my cleaning table over? i snap awake at some dawn. the patrons researching the blues instructor's theories around me, then give me a glass of jim beam. boom! the business group soon concludes. caught one miracle? survive the tedious rituals you refuse to provide? whatever happens, i throw shit in your face. joy ries away? you could give acapindulgent half your life away, though i am nothing especially sexy. this sift, this being sveng? do impotent and desperate oozing drug as i shake and shine? you adored me. brains run out when people charitable that into you. 

Now it was time to make some cleaning.

## Heuristic filters

The next big thing to do was filter some really good ones from this endless flow of the text.

At first, I made a script with some simple heuristic filters such as:
  - reject a creation of new, non-existing words,
  - reject phrases with two unconnected verbs in a row,
  - reject phrases with several duplicating words,
  - reject phrases with no punctuation or with too many punctuation marks.

The application of this script cut the initial text flow into a subsequence of valid chunks.

> a slave has no more say in his language but he hasn't to speak out!
>
> the doll has a variety of languages, so its feelings have to fill up some time of the day - to - day journals.
> the doll is used only when he remains private.
> and it is always effective.
>
> leave him with his monk - like body.
>
> a little of technique on can be helpful.
>
> out of his passions remain in embarrassment and never wake.
>
> adolescence is the university of manchester.
> the senior class of manchester... the senior class of manchester.

## Critic subsystem

At last, I trained the Critic subsystem.
This neural network uses a [BERT](https://github.com/google-research/bert) architecture implemented again by [huggingface](https://github.com/huggingface/transformers). Again I took a public available pre-trained network model and finetuned it on my labeled 1K chunks dataset to predict the label of any given chunk.

Here I used manual labeling of these chunks with two classes, GOOD/BAD. Most of the labeling was done by a friend of mine, Ivan [@kr0niker](https://www.yamshchikov.info/) Yamshchikov, and some I did myself. We marked a chunk as BAD in case it was grammatically incorrect or just too boring or too stupid. Overall, I used approx 1K of labeled chunks, balanced (one half of them were GOOD, the other half -- BAD).

Finally, I made a pipeline that includes the Generator subsystem, some heuristic filters, and the Critic subsystem.
Here it is a short sample of the final results:

> a sudden feeling of austin lemons, a gentle stab of disgust.
> i'm what i'm.
> 
> humans whirl in night and distance.
> 
> by the wonders of them.
> 
> we shall never suffer this.
> if the human race came along tomorrow, none of us would be as wise as they already would have been.
> there is a beginning and an end.
> 
> both of our grandparents and brothers are overdue.
> he either can not agree or he can look for someone to blame for his death.
> 
> he has reappeared from the world of revenge, revenge, separation, hatred.
> he has ceased all who have offended him.
> 
> he is the one who can remember that nothing remotely resembles the trip begun in retrospect.
> what's up?
> 
> and i don't want the truth.
> not for an hour.

[The huge blob of generated text could be found here](https://github.com/altsoph/paranoid_transforner/blob/master/NaNoGenMo_50K_words_sample.txt).

## Code overview

Here is a short description of scripts from this project:
- gpt1tokenize_trainset.py -- used to tokenize the fine-tuning dataset and add the conditioning labels
- gpt1finetune.py -- used to fine-tune the Generator network on the prepared dataset
- gpt1sample.py -- used to sample texts from the Generator network

- simple_cleaner.py -- holds the heuristic filters

- train_classifier.py -- used to train the BERT-based classifier (Critic)
- do_critic.py -- applies Critic to the samples
- weight_samples.py + cleaner_on_bert_weights.py -- used to filter samples based on Critic scores


## Nervous handwriting

As much as the resulting text basically reminded me of neurotic/paranoid notes I decided to use this effect and make it deeper:

I took an [implementation by Sean Vasquez](https://github.com/sjvasquez/handwriting-synthesis) of the handwriting synthesis experiments from the paper [Generating Sequences with Recurrent Neural Networks by Alex Graves](https://arxiv.org/abs/1308.0850) and patched it a little. Specifically, I used a bias parameter to make the handwriting shakiness depended on the sentiment strength of a given sentence.

Take a look at the example:

<img src="https://raw.githubusercontent.com/altsoph/paranoid_transformer/master/handwriting.png" alt="drawing" />

## Freehand drawings

At some point, I realized that this diary lacks freehand drawings, so I decided to add some. I used my modification of a [pytorch implementation](https://github.com/alexis-jacq/Pytorch-Sketch-RNN) of [arXiv:1704.03477](https://arxiv.org/abs/1704.03477) trained on 
[Quick, Draw! Dataset](https://github.com/googlecreativelab/quickdraw-dataset). Each time any of categories from the dataset appears on the page I generate and add random picture somewhere arround.

<img src="https://raw.githubusercontent.com/altsoph/paranoid_transformer/master/pics_samples.png" alt="drawing" /> 

## Covers and PDF compilation

I drew some covers and used the [rsvg-convert library](https://en.wikipedia.org/wiki/Librsvg) to build a PDF file from separate pages in SVG.

Covers:

<img src="https://raw.githubusercontent.com/altsoph/paranoid_transformer/master/paranoid_transformer.png" alt="drawing" width="300"/> <img src="https://raw.githubusercontent.com/altsoph/paranoid_transformer/master/paranoid_transformer_back.png" alt="drawing" width="300"/>

The resulting diary (40 Mb):
https://github.com/altsoph/paranoid_transformer/raw/master/paranoid_transformer_w_pics.pdf

## Papers, publications, releases, links

* [ICCC 2020 Proceedings, P.146-152](http://computationalcreativity.net/iccc20/wp-content/uploads/2020/09/ICCC20_Proceedings.pdf): Paranoid Transformer. Yana Agafonova, Alexey Tikhonov and Ivan Yamshchikov
* Future Internet Journal: [Paranoid Transformer: Reading Narrative of Madness as Computational Approach to Creativity](https://www.mdpi.com/1999-5903/12/11/182/htm)
* [Pre-oder the book](https://deadalivemagazine.com/press/paranoid-transformer.html)

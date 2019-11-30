import sys
from nltk.tokenize import sent_tokenize, word_tokenize
from collections import defaultdict
from nltk import pos_tag

vocab = set()
for line in open("vocab.txt", encoding='utf-8'):
	vocab.add( line.strip() )

outfh = open(sys.argv[2], "w", encoding='utf-8')

lines_cnt = sentences_cnt = 0
cases = defaultdict(int)
for ln, line in enumerate(open(sys.argv[1], encoding='utf-8')):
	if line[0] == '=': continue
	lines_cnt += 1

	sents = sent_tokenize(line)
	sentences_cnt += len(sents)
	print(file=outfh)
	for sent in sents:
		words = word_tokenize(sent)
		tmp = sent.replace('#',' ').replace('...','#').replace('!','#').replace('?','#').replace(',','#').replace('--','#').replace('-','#').replace(';','#')\
			      .replace(':','#').replace('`','#').replace('"','#').replace('.','#').replace(' ','')

		no_punct = []
		size = 0
		for npidx,w in enumerate(words):
			if w in ('...','!','?',',','--','-',';',':','`','"','.'):
				no_punct.append(size)
				size = 0
			else:
				size += 1
		no_punct.append(size)

		pos = pos_tag(words)
		skip = False
		for idx,(w,p) in enumerate(pos[:-1]):
			# VB verb, base form take
			# VBD verb, past tense took
			# VBG verb, gerund/present participle taking
			# VBN verb, past participle taken
			# VBP verb, sing. present, non-3d take
			# VBZ verb, 3rd person sing. present takes			
			if p in ('VB','VBD','VBG','VBN','VBP','VBZ') and pos[idx+1][1] in ('VB','VBD','VBG','VBN','VBP','VBZ'):
				if p == 'VBD' and pos[idx+1][1] == 'VBN': continue
				if p == 'VB' and pos[idx+1][1] == 'VBN': continue
				if p == 'VBP' and pos[idx+1][1] == 'VBN': continue
				if p == 'VBZ' and pos[idx+1][1] == 'VBN': continue
				if w == 'been' and pos[idx+1][1] == 'VBN': continue
				if w in ('be','was','are','is',"'re","'s","been","have") and pos[idx+1][1] == 'VBG': continue
				if w == 'i': continue
				# print('VERB', (w,p), pos[idx+1], sent)
				cases['verbverb'] += 1
				skip = True
				break
		# it's bad if several verbs in a row
		if set(words)-vocab:
			cases['new_word'] += 1
			skip = True
		elif max(no_punct)>25:
			cases['no_punct'] += 1
			skip = True
		elif len(words)>=60:
			cases['to_long'] += 1
			skip = True
		elif "###" in tmp:
			cases['manypuncts'] += 1
			skip = True
		for idx,w in enumerate(words[:-1]):
			if w == words[idx+1]:
				cases['duplicate_words'] += 1
				skip = True
				break
		if sent[-1] not in '.!?':
			cases['badend'] += 1
			skip = True

		if skip:
			print(file=outfh)
		else:
			print(sent,file=outfh)

	print(lines_cnt, sentences_cnt, cases.items())

outfh.close()

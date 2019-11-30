import sys

ofh = open(sys.argv[2], 'w', encoding='utf-8')

prev_blank = True
for ln, line in enumerate(open(sys.argv[1], encoding='utf-8')):
	text,_,_,score = line.strip().split('\t')

	if text == '----------' or float(score)<0.9:
		if not prev_blank:
			print(file=ofh)
		prev_blank = True
	else:
		print(text,file=ofh)
		prev_blank = False

ofh.close()
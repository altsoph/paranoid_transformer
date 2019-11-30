import sys
import math 

collected = []

def sm(x):
	a = math.exp(x[1])
	b = math.exp(x[2])
	return b/(a+b+1e-8)

ofh = open(sys.argv[2], 'w', encoding='utf-8')

for line in open(sys.argv[1], encoding='utf-8'):
	chunks = line.strip().split('\t')
	chunks[1] = float(chunks[1])
	chunks[2] = float(chunks[2])
	chunks.append( sm(chunks) )
	print("\t".join(map(str,chunks)),file=ofh)

ofh.close()

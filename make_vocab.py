import spacy

#Loading English
spacy_de = spacy.load("de_core_news_sm")

def tokenize_de(text):
	return [tok.text for tok in spacy_de.tokenizer(text)]
	
def build_vocab(path):

	files = open(path, 'r').read().splitlines()
	captions = []
	for cap in files:
		captions.append(cap.split('|')[-1])
	
	#Beginning of Sentence
	BOS_WORD = '<s>'
	#End of Sentence
	EOS_WORD = '</s>'
	#Padding
	BLANK_WORD = "<blank>"
	
	dict_word_int = {}
	#Creating a Dictionary in the format {Token : Integer}
	i=0
	dict_word_int.update({BOS_WORD:i})
	i+=1
	dict_word_int.update({EOS_WORD:i})
	i+=1
	dict_word_int.update({BLANK_WORD:i})
	i+=1

	for cap in captions:
		for word in tokenize_de(cap):
			if word not in dict_word_int.keys():
				dict_word_int.update({word:i})
				i+=1
	dict_int_word = {value : key for (key, value) in dict_word_int.items()}
	return dict_word_int, dict_int_word	

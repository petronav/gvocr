#from word2number import w2n
from difflib import SequenceMatcher
from babel.numbers import format_currency
from num2words import num2words
import re

def check_num_pres(input_str):
    return any(char.isdigit() for char in input_str)

def check_string_similarity(str1, str2):
    if not(check_num_pres(str1) and check_num_pres(str2)):
        return SequenceMatcher(a=str1.lower(), b=str2.lower()).ratio() > 0.80
def spellcorrection(word):
    replace_words_list = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
		"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "and",
		"sixteen", "seventeen", "eighteen", "nineteen","twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety",
		"hundred", "thousand", "lakh", "crore"]
    for replace_word in replace_words_list:
        if check_string_similarity(word, replace_word):
            try:
                word = replace_word
            except Exception as e:
                pass
    return word
def onlyalphabetic(string1):
    string2 = re.sub('[^A-Za-z\s]+', '', string1)
    return string2

def word2num(textnum, numwords={}):
	textnum = onlyalphabetic(textnum)
	textnum = textnum.lower()
	print(textnum)
	if not numwords:
		units = [
		"zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
		"nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
		"sixteen", "seventeen", "eighteen", "nineteen", 
		]

		tens_dict = {"twenty" : 20, "thirty" : 30, "forty" : 40, "fourty" : 40, "fifty" : 50, "sixty" : 60, "seventy" : 70, "eighty" : 80, "ninety" : 90}
		scales_dict = {"hundred" : 100, "thousand" : 1000, "lakh" : 100000, "crore" : 10000000}

		numwords["and"] = (1, 0)
		for idx, word in enumerate(units):    numwords[word] = (1, idx)
		for k,v in tens_dict.items():
			numwords[k] = (1, v)
		for k,v in scales_dict.items():
			numwords[k] = (v,0)
	current = result = 0
	for word in textnum.split():
		if word not in numwords:
			word = spellcorrection(word)
		try:
			scale, increment = numwords[word]
			current = current * scale + increment
			if scale > 100:
				result += current
				current = 0
		except Exception as e:
			continue
	ret_res_num = result + current
	print(ret_res_num)
	print(format_currency(ret_res_num, 'INR', locale='en_IN').split()[-1])
	return format_currency(ret_res_num, 'INR', locale='en_IN').split()[-1]

word2num('Thirteen Thousand Forty Six')

def num2word(numstr):
	numstr = str(numstr)
	num = numstr.replace(",","")
	dot_pres = num.find(".")
	if dot_pres != -1:
		num = num[:dot_pres]
	#print("num : ", num)
	digits_check = re.findall(r'\d', num)
	if len(digits_check) >0:		
		a = num2words(num, to='cardinal', lang='en_IN')
		a = a.replace(",","")
		a = a.replace("-"," ")
		a = a[:a.find(" point")]
		#print(a)
		return a
	else:
		#print("vacant")
		return "v"

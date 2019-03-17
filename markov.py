import collections

#data = "i have a note book . how are you . wubba lubba dub dub . i have a pendrive . you have two pens . you and i .  i dont know why i am doing this . i dont i dont "

data = " A Pubg Tournament . A Mumbai Hackathon . An Amazing Mumbai Hacking Tournament . Amazing #MumbaiHackathon2019 . Amazing Night . A Mumbai Breakfast ."


probabilties = {}


def prob(probabilties, key, value):
	if key not in probabilties:
		probabilties[key] = []
	probabilties[key].append(value)

def markov():
	tokens = data.strip().split(" ")
	for i in range(len(tokens)-1):
		prob(probabilties, tokens[i], tokens[i+1])

def calculate_predictions(probabilties):
	for key in probabilties:
		probabilties[key] = collections.Counter(probabilties[key]).most_common()
		probabilties[key] = [i[0] for i in probabilties[key]]




def markov_here(t):
	global probabilties
	markov()
	calculate_predictions(probabilties)
	#print(probabilties)

	if t in probabilties:
		top = probabilties[t][:3]
		return top
		#print("----->",top,)


	# if t == '1':
	# 	prediction[]
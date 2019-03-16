data = " i like fruits . i am in mumbai hackathon . what is this . who is markov and did he invent this chain . if you pull chances are you might be finned ."

def tokenizer(data):
	return data.strip().split(" ")

possible_dic, probability = {}, {}

def possibilitizer(next,key):
	if key not in possible_dic:
		possible_dic[key] = []
	possible_dic[key].append(next)


tokens = tokenizer(data)
for i in range(len(tokens)- 1):
	possibilitizer(tokens[i+1], tokens[i])
print(possible_dic)
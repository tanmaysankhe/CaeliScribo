import collections


class Markov:

	def __init__(self):
		self.probabilties = {}
		self.data = " A Pubg Tournament . A Mumbai Hackathon . An Amazing Mumbai Hacking Tournament . Amazing #MumbaiHackathon2019 . Amazing Night . A Mumbai Breakfast ."
		self.markov()
		self.calculate_predictions(self.probabilties)

	def prob(self, probabilties, key, value):
		if key not in self.probabilties:
			self.probabilties[key] = []
		self.probabilties[key].append(value)

	def markov(self):
		tokens = self.data.strip().split(" ")
		for i in range(len(tokens)):
			if len(tokens[i]) > 1:
				self.prob(self.probabilties, tokens[i][:1], tokens[i][1:])
			elif len(tokens[i]) > 2:
				self.prob(self.probabilties, tokens[i][:2], tokens[i][2:])
			elif len(tokens[i]) > 3:
				self.prob(self.probabilties, tokens[i][:3], tokens[i][3:])

	def calculate_predictions(self,probabilties):
		for key in probabilties:
			self.probabilties[key] = collections.Counter(probabilties[key]).most_common()
			self.probabilties[key] = [i[0] for i in probabilties[key]]




	def markov_here(self,t):
		print(self.probabilties)
		self.calculate_predictions(self.probabilties)
		if t in self.probabilties:
			top = self.probabilties[t][:3]
			return top

import os
from tqdm import tqdm
import json
import sys
import numpy as np

class Preprocessor(object):
	"""
	This is the class that will perform all of the necessary preprocessing on
	the data.

	Example of usage:

		p = Preprocessor("leagueoflegends", 1e7, 75)
		comments = p.process()
		out = open('out.txt', 'w+', encoding='utf-8')
		for c in comments:
			out.write(c[0])
		out.close()
		good = p.statistics(comments)

	The constructor has arguments:
	 - target_subreddit: A string that represents the subreddit we wish to
	   focus. If it is none, then we collect data on every subreddit, which
	   we do not recommend at the moment.
	 - break_limit: An int that represents the number of comments we wish to
	   collect before we stop.
	 - threshold: The percentile that we wish to extract comments from. For
	   example, if this value is 90, then only extract comments in the upper
	   10 percent of rated comments.
	"""
	def __init__(self, target_subreddit, break_limit, threshold, custom_file="",
		custom=False
	):
		self.target_subreddit = target_subreddit
		self.break_limit = break_limit
		self.threshold = threshold

	"""
	Performs the file reading that transforms our intended dataset into the
	form: [(comment, score), ...]
	Uses the tqdm library to create a progress bar for sanity purposes.
	Inputs:
	 - custom_file: A string denoting the custom file that we would like to use.
	   Only considered if custom=True.
	 - custom: A boolean denoting whether or not we would like to use a custom
	   file.
	"""
	def process(self, custom_file="", custom=False):
		print("Processing file...")

		#Default input file
		f_name = '../RC_2015'
		if custom:
			f_name = custom_file

		try:
			f = open(f_name, 'r')
		except:
			print("File not found!")
			sys.exit()

		line_count = 0
		sizecounter = 0
		sizecounter += os.stat(f_name).st_size
		output = []
		#Initialize progress bar
		with tqdm(total=sizecounter,
				unit='B', unit_scale=True, unit_divisor=1024) as pbar:
			with open(f_name, 'r', encoding="utf-8") as fh:
				"""
				If we wish to use a custom file, 
				"""
				if custom:
					"""
					We simply add each line of the input file to a custom
					structure.
					"""
					for line in fh:
						output.append(line)
						if line:
							pbar.set_postfix(file=f_name[-10:], refresh=False)
							pbar.update(sys.getsizeof(line))
				else:
					"""
					For each file in the input file, we load each comment and 
					see if the comment that we observe is the same as the one
					we're looking for. If so, we record the body of the comment
					and its core.
					"""
					for line in fh:
						comment = json.loads(line)
						comment_as_dict = dict(comment)
						subreddit = comment_as_dict['subreddit']
						if subreddit == self.target_subreddit:
							score = int(comment_as_dict['ups'])
							output.append((comment_as_dict['body'], score))
						line_count += 1
						"""
						We stop after some number of comments if we are not
						looking at every single comment.
						"""
						if self.break_limit != None:
							if line_count > self.break_limit:
								break
						if line:
							pbar.set_postfix(file=f_name[-10:], refresh=False)
							pbar.update(sys.getsizeof(line))
		f.close()

		print("Finished processing")
		return output, line_count

	"""
	Takes in the original comment base and returns a subset of of comments that
	pertain to the percentile requirements that we've previously set forth.
	"""
	def statistics(self, comments):
		"""
		We first extract the scores and compute the lower bound of the
		percentile.
		"""
		scores = []
		for comment in comments:
			scores.append(comment[1])
		scores = np.array(scores)
		lower_bound = np.percentile(scores, self.threshold)
		"""
		We then make another pass through the comments and only save the comments
		that are above the bound.
		"""
		good_comments = []
		for comment in comments:
			if comment[1] > lower_bound:
				good_comments.append(comment)
		return good_comments
		"""
		This is pretty inefficient but in practice, it runs pretty quickly.
		"""


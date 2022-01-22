import argparse
import pandas as pd
import numpy as np
import random

def read_23andme_genome(filename):
	colnames = ['rsid', 'chromosome', 'position', 'genotype']
	df = pd.read_csv(filename, sep='\t', skiprows=20, names=colnames, low_memory=False)
	return df


def randomly_pass_down(alleles):
	""" Input is two bases: AA, AT, GC, etc """
	if random.random() > 0.5:
		return alleles[1]
	else:
		return alleles[0]


def missing_gene_fallback(snp_id):
	data = {
		'rs6828137': ('G', 0.55, 'T'),
		'rs2748901': ('G', 0.43, 'A'),
		'rs147068120': ('C', 0.96, 'T'),
		'rs796296176': ('D', 0.996, 'I'),
		'rs11957757': ('G', 0.77, 'A'), # Latin American 2
		'rs12552712': ('T', 0.67, 'C'), # Latin American 2
		'rs6944702': ('T', 0.51, 'C'), # European
		'rs13297008': ('G', 0.37, 'A'), # European
		'rs2854746': ('G', 1.0, 'C'), # Latin American 2
		'rs116359091': ('G', 0.978, 'A'), # European
	}
	ref, p_ref, alt = data[snp_id]
	if random.random() < p_ref:
		return ref
	else:
		return alt


def breed(genome1, genome2):
	"""
	Inputs: two dictionaries {rs123123: AA, rs5535: GG, etc}
	"""
	child_genome = {}

	for snp_id in set(genome1).union(set(genome2)):
		try:
			parent1_allele = randomly_pass_down(genome1[snp_id])
		except KeyError:
			parent1_allele = missing_gene_fallback(snp_id)
		try:
			parent2_allele = randomly_pass_down(genome2[snp_id])
		except KeyError:
			parent2_allele = missing_gene_fallback(snp_id)


		child_alleles = parent1_allele + parent2_allele
		child_genome[snp_id] = child_alleles

	return child_genome


def predict_eye_color(genome, irisplex):
	"""
	Inputs:
		- genome is a dict {'rs234234': 'AA', ...}
		- irisplex is a pandas df with index 'rsid'
	"""
	irisplex2 = irisplex.copy()
	irisplex2['my_genome'] = pd.Series(genome)
	irisplex2 = irisplex2.dropna()

	allele1 = irisplex2['my_genome'].str[0]
	allele2 = irisplex2['my_genome'].str[1]
	allele1_is_minor = (allele1 == irisplex2['minor_allele']).astype(int)
	allele2_is_minor = (allele2 == irisplex2['minor_allele']).astype(int)
	minor_allele_count = allele1_is_minor + allele2_is_minor
	
	beta1_x = irisplex2['beta1'] * minor_allele_count
	beta2_x = irisplex2['beta2'] * minor_allele_count
	
	ALPHA1 = 3.94
	ALPHA2 = 0.65
	a = np.exp(ALPHA1 + sum(beta1_x))
	b = np.exp(ALPHA2 + sum(beta2_x))

	p_blue = a / (1 + a + b)
	p_other = b / (1 + a + b)
	p_brown = 1 - p_blue - p_other
	return p_blue, p_other, p_brown


def predict_eye_color_2(genome, abd):
	j = abd.copy()
	j['genotype'] = pd.Series(genome)
	j = j.dropna()

	allele1 = j['genotype'].str[0]
	allele2 = j['genotype'].str[1]
	j['alt_allele_count'] = (allele1==j['Alt allele']).astype(int) + (allele2==j['Alt allele']).astype(int)
	
	effect = j['alt_allele_count'] * j['Beta']
	
	return 1 - sum(effect)


def predict_hair_color(genome, hirisplex):
	h2 = hirisplex.copy()
	h2['my_genome'] = pd.Series(genome)
	h2 = h2.dropna()

	allele1 = h2['my_genome'].str[0]
	allele2 = h2['my_genome'].str[1]
	allele1_is_minor = (allele1 == h2['minor_allele']).astype(int)
	allele2_is_minor = (allele2 == h2['minor_allele']).astype(int)
	minor_allele_count = allele1_is_minor + allele2_is_minor
	
	brown_exp = np.exp(sum(h2['effect_brown'] * minor_allele_count) - 2.0769)
	red_exp = np.exp(sum(h2['effect_red'] * minor_allele_count) - 6.3953)
	black_exp = np.exp(sum(h2['effect_black'] * minor_allele_count) - 2.4029)
	
	sum_exp = brown_exp + red_exp + black_exp

	p_brown = brown_exp / (1 + sum_exp)
	p_red = red_exp / (1 + sum_exp)
	p_black = black_exp / (1 + sum_exp)
	p_blond = 1 - p_brown - p_red - p_black

	return p_brown, p_red, p_black, p_blond


def print_parent(name, p_eyes1, p_eyes2, p_hair):
	p_brnhr, p_red, p_black, p_blond = p_hair
	print(f'{name}')
	p_blue, p_other, p_brown = p_eyes1
	print((f'\tEyes (method 1): blue {round(p_blue*100)}%, '
		   f'grn/hzl {round(p_other*100)}%, '
		   f'brown {round(p_brown*100)}%'))
	print(f'\tEye darkness (method 2): {round(p_eyes2)} out of 6')
	print((f'\tHair: blond {round(p_blond*100)}%, '
		   f'red {round(p_red*100)}%, '
		   f'brown {round(p_brnhr*100)}%, '
		   f'black {round(p_black*100)}%'))


def main(filenames):
	if len(filenames) != 2:
		raise Exception('Must give exactly 2 filenames (one for each parent)')

	# Read 23 and me data
	filename1, filename2 = filenames
	df1 = read_23andme_genome(filename1)
	df2 = read_23andme_genome(filename2)

	# Read model data
	irisplex = pd.read_csv('data/irisplex.csv').set_index('rsid')
	hirisplex = pd.read_csv('data/hirisplex.csv').set_index('rsid')
	abd = pd.read_csv("data/abd1239t1.csv").set_index('RS')

	# Take only genes of interest
	my_rsid = set(hirisplex.index).union(set(irisplex.index)).union(set(abd.index))
	df1_iris = df1[df1['rsid'].isin(my_rsid)]
	df2_iris = df2[df2['rsid'].isin(my_rsid)]

	# Parent genomes
	parent1_genome = df1_iris.set_index('rsid')['genotype'].to_dict()
	parent2_genome = df2_iris.set_index('rsid')['genotype'].to_dict()
	# print('intersection:', set(parent1_genome).intersection(set(parent2_genome)))
	# print('diffs:', set(parent1_genome).union(set(parent2_genome)) - set(parent1_genome).intersection(set(parent2_genome)))
	# print(pd.concat([df1_iris.set_index('rsid')['genotype'], df2_iris.set_index('rsid')['genotype']], axis=1))
	p_eyes1 = predict_eye_color(parent1_genome, irisplex)
	p_eyes2 = predict_eye_color_2(parent1_genome, abd)
	p_hair = predict_hair_color(parent1_genome, hirisplex)
	print_parent('Parent 1', p_eyes1, p_eyes2, p_hair)
	p_eyes1 = predict_eye_color(parent2_genome, irisplex)
	p_eyes2 = predict_eye_color_2(parent2_genome, abd)
	p_hair = predict_hair_color(parent2_genome, hirisplex)
	print_parent('Parent 2', p_eyes1, p_eyes2, p_hair)

	# Get all offspring possibilities
	n = 200
	children_eyes = []
	children_hair = []
	for i in range(n):
		child_genome = breed(parent1_genome, parent2_genome)
		x = random.random()

		# Eye color
		# p_blue, p_other, p_brown = predict_eye_color(child_genome, irisplex)
		# if x < p_blue:
		# 	children_eyes.append('blue')
		# if p_blue <= x < p_blue+p_brown:
		# 	children_eyes.append('brown')
		# if p_blue+p_brown <= x:
		# 	children_eyes.append('other')
		pred_darkness = predict_eye_color_2(child_genome, abd)
		children_eyes.append(round(pred_darkness))

		# Hair color
		p_brnhr, p_red, p_black, p_blond = predict_hair_color(child_genome, hirisplex)
		if x < p_brnhr:
			children_hair.append('brown')
		if p_brnhr <= x < p_brnhr+p_black:
			children_hair.append('black')
		if p_brnhr+p_black <= x:
			children_hair.append('blond')


	# n_blue = children_eyes.count('blue')
	# n_other = children_eyes.count('other')
	# n_brown = children_eyes.count('brown')
	print(f'Randomly generated {n} children.')
	print('Eyes:')
	for i in range(1, 7):
		print(f'\t{i}: {children_eyes.count(i)}')
	print('Hair:')
	print(f'\tBlond: {children_hair.count("blond")}')
	print(f'\tBrown: {children_hair.count("brown")}')
	print(f'\tBlack: {children_hair.count("black")}')
	# if n <= 12:
	# 	s = ''
	# 	for i in range(n_blue):
	# 		s += '🔵'
	# 	for i in range(n_other):
	# 		s += '🟡'
	# 	for i in range(n_brown):
	# 		s += '🟤'
	# 	print(s)
	# else:
	# 	print('Hair:')
	# 	print(f'\tBlond: {children_hair.count("blond")}')
	# 	print(f'\tBrown: {children_hair.count("brown")}')
	# 	print(f'\tBlack: {children_hair.count("black")}')
	# 	print('Eyes:')
	# 	print(f'\tBlue: {n_blue}')
	# 	print(f'\tGreen/Hazel: {n_other}')
	# 	print(f'\tBrown: {n_brown}')


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument(
		'-f',
		'--filenames', 
		help='PDF filenames to compare', 
		required=True,
		nargs='+',
	)
	args = parser.parse_args()

	main(filenames=args.filenames)
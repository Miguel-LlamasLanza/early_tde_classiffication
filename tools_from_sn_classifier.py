#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:17:45 2023

@author: lmiguel
"""
import pandas as pd
import numpy as np
from fink_sn_AL_classifier.actsnfink import classifier_sigmoid
from fink_sn_AL_classifier.actsnfink import sigmoid
from fink_sn_AL_classifier.actsnfink.early_sn_classifier import mag2fluxcal_snana
import matplotlib.pyplot as plt

# This contains tools exracted from the early sn classifier github, addapted for our use
# https://github.com/emilleishida/fink_sn_activelearning/tree/master


def convert_full_dataset(pdf: pd.DataFrame, obj_id_header='candid'):
	"""Convert an entire data set from mag to fluxcal.

	Parameters
	----------
	pdf: pd.DataFrame
		DataFrame with same format as the parquet files.
	obj_id_header: str (optional)
		Object identifier. Options are ['objectId', 'candid'].
		Default is 'candid'.

	Returns
	-------
	pd.DataFrame
		Columns are ['objectId', 'type', 'MJD', 'FLT',
		'FLUXCAL', 'FLUXCALERR'].
	"""
	# Ia types in TNS
	Ia_group = ['SN Ia', 'SN Ia-91T-like', 'SN Ia-91bg-like', 'SN Ia-CSM',
				'SN Ia-pec', 'SN Iax[02cx-like]']

	# hard code ZTF filters
	filters = ['g', 'r']

	lc_flux_sig = []

	for index in range(pdf.shape[0]):
		print('\r Objects converted: {}/{}'.format(index + 1, pdf.shape[0]), end='\r')

		name = pdf[obj_id_header].values[index]

		try:
			sntype_orig = pdf['TNS'].values[index]
			if sntype_orig == -99:
				sntype_orig = pdf['cdsxmatch'].values[index]

			if sntype_orig in Ia_group:
				transient_type = 'Ia'
			else:
				transient_type = str(sntype_orig).replace(" ", "")
		except KeyError:
			transient_type = 'TDE'

		for f in range(1,3):

			if isinstance(pdf.iloc[index]['fid'], str):
				ffs = np.array([int(item) for item in pdf.iloc[index]['fid'][1:-1].split()])
				filter_flag = ffs == f
				mjd = np.array([float(item) for item in pdf.iloc[index]['jd'][1:-1].split()])[filter_flag]
				mag = np.array([float(item) for item in pdf.iloc[index]['magpsf'][1:-1].split()])[filter_flag]
				magerr = np.array([float(item) for item in pdf.iloc[index]['sigmapsf'][1:-1].split()])[filter_flag]
			else:
				filter_flag = pdf['fid'].values[index] == f
				mjd = pdf['jd'].values[index][filter_flag]
				mag = pdf['magpsf'].values[index][filter_flag]
				magerr = pdf['sigmapsf'].values[index][filter_flag]

			fluxcal = []
			fluxcal_err = []
			for k in range(len(mag)):
				f1, f1err = mag2fluxcal_snana(mag[k], magerr[k])
				fluxcal.append(f1)
				fluxcal_err.append(f1err)

			for i in range(len(fluxcal)):
				lc_flux_sig.append([name, transient_type, mjd[i], filters[f - 1],
									fluxcal[i], fluxcal_err[i]])

	lc_flux_sig = pd.DataFrame(lc_flux_sig, columns=['id', 'type', 'MJD',
													 'FLT', 'FLUXCAL',
													 'FLUXCALERR'])

	return lc_flux_sig



def featurize_full_dataset(lc: pd.DataFrame, screen=False):
	"""Get complete feature matrix for all objects in the data set.

	Parameters
	----------
	lc: pd.DataFrame
		Columns should be: ['objectId', 'type', 'MJD', 'FLT',
		'FLUXCAL', 'FLUXCALERR'].
	screen: bool (optional)
		If True print on screen the index of light curve being fit.
		Default is False.

	Returns
	-------
	pd.DataFrame
		Features for all objects in the data set. Columns are:
		['objectId', 'type', 'a_g', 'b_g', 'c_g', 'snratio_g',
		'mse_g', 'nrise_g', 'a_r', 'b_r', 'c_r', 'snratio_r',
		'mse_r', 'nrise_r']
	"""

	# columns in output data matrix
	columns = ['id', 'type', 'a_g', 'b_g', 'c_g',
			   'snratio_g', 'mse_g', 'nrise_g', 'a_r', 'b_r', 'c_r',
			   'snratio_r', 'mse_r', 'nrise_r']

	features_all = []

	for indx in range(np.unique(lc['id'].values).shape[0]):

		if screen:
			print('\r Objects processed: {}'.format(indx + 1), end='\r')

		name = np.unique(lc['id'].values)[indx]

		obj_flag = lc['id'].values == name
		sntype = lc[obj_flag].iloc[0]['type']

		line = [name, sntype]

		features = classifier_sigmoid.get_sigmoid_features_dev(lc[obj_flag][['MJD',
														  'FLT',
														  'FLUXCAL',
														  'FLUXCALERR']])

		if screen:
			# Plot LC and sigmoid fit
			# features for different filters
			a = {}
			b = {}
			c = {}
			snratio = {}
			mse = {}
			chisq = {}
			nrise = {}

			# Get from output
			[a['g'], b['g'], c['g'], snratio['g'], mse['g'], nrise['g'],
			a['r'], b['r'], c['r'], snratio['r'], mse['r'], nrise['r'], chisq['g'], chisq['r']
				] = features

			plt.figure()
			# ax = plt.gca()
			plt.title(name)
			plt.xlabel('Time (days)')
			plt.ylabel('Fluxcal')
			# ax.text(0.1, 0.9, 'b_g: {}\n b_r: {}'.format(b['g'], b['r'], transform=ax.transAxes))
			for filt, color in zip(['g', 'r'], ['#15284F', '#F5622E']):

				# mask_filt = mask[filt]
				masked_lc = lc[(obj_flag) & (lc['FLT']==filt)]

				if len(masked_lc)!=0:

					# t0 = masked_lc['MJD'][masked_lc['FLUXCAL'].idxmin()]
					t0 = min(masked_lc['MJD'])
					tmax = masked_lc['MJD'][masked_lc['FLUXCAL'].idxmax()]

					# t0 = time[flux.argmax()] - time[0]

					x = np.linspace(t0 - tmax - 20, tmax - t0 + 30, num = 200)

					sigmoid_fit = sigmoid.fsigmoid(x, a[filt], b[filt], c[filt])

					plt.plot(x, sigmoid_fit, c = color)
					plt.errorbar(masked_lc['MJD'] - t0 , masked_lc['FLUXCAL'], fmt = '.', elinewidth=0.5,
								  yerr = masked_lc['FLUXCALERR'], label = filt, c = color)
					plt.legend()


		for j in range(len(features)):
			line.append(features[j])

		features_all.append(line)

	columns.extend(['chisq_g', 'chisq_r'])
	feature_matrix = pd.DataFrame(features_all, columns=columns)

	return feature_matrix


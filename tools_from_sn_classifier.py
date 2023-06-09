#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:17:45 2023

@author: lmiguel
"""
import pandas as pd
import numpy as np
from Data_other_studies.fink_sn_activelearning.actsnfink import classifier_sigmoid
import fit_Lightcurve as fit_lc
import matplotlib.pyplot as plt

# This contains tools exracted from the early sn classifier github, addapted for our use
# https://github.com/emilleishida/fink_sn_activelearning/tree/master



def mag2fluxcal_snana(magpsf: float, sigmapsf: float):
	""" Conversion from magnitude to Fluxcal from SNANA manual
	Parameters
	----------
	magpsf: float
		PSF-fit magnitude from ZTF.
	sigmapsf: float
		Error on PSF-fit magnitude from ZTF.

	Returns
	----------
	fluxcal: float
		Flux cal as used by SNANA
	fluxcal_err: float
		Absolute error on fluxcal (the derivative has a minus sign)
	"""
	if magpsf is None:
		return None, None
	fluxcal = 10 ** (-0.4 * magpsf) * 10 ** (11)
	fluxcal_err = 9.21034 * 10 ** 10 * np.exp(-0.921034 * magpsf) * sigmapsf

	return fluxcal, fluxcal_err

def convert_full_dataset(pdf: pd.DataFrame, obj_id_header='candid'):
	"""Convert an entire data set from mag to fluxcal.

	Parameters
	----------
	pdf: pd.DataFrame
		Read directly from parquet files.
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
			print('indx: ', indx)

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
			nrise = {}

			# Get from output
			[a['g'], b['g'], c['g'], snratio['g'], mse['g'], nrise['g'],
			a['r'], b['r'], c['r'], snratio['r'], mse['r'], nrise['r']] = features

			plt.figure()
			plt.title(name)
			for filt in ['g', 'r']:

				# mask_filt = mask[filt]
				masked_lc = lc[obj_flag][lc['FLT']==filt].copy()

				if len(masked_lc)!=0:

					t0 = masked_lc['MJD'][masked_lc['FLUXCAL'].idxmin()]
					tmax = masked_lc['MJD'][masked_lc['FLUXCAL'].idxmax()]

					# t0 = time[flux.argmax()] - time[0]

					x = np.linspace(-60, 60, num = 200)

					sigmoid = fit_lc.sigmoid_profile(x, a[filt], b[filt], c[filt])

					plt.plot(x, sigmoid)
					plt.scatter(masked_lc['MJD'] - t0 , masked_lc['FLUXCAL'])


		for j in range(len(features)):
			line.append(features[j])

		features_all.append(line)

	feature_matrix = pd.DataFrame(features_all, columns=columns)

	return feature_matrix

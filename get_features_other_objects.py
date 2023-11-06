#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:20:40 2023

@author: lmiguel
"""

import pandas as pd
import os
import glob
import logging
import numpy as np
import matplotlib.pyplot as plt
from fink_sn_AL_classifier.actsnfink.early_sn_classifier import mag2fluxcal_snana
from light_curve.light_curve_py import RainbowFit


logging.basicConfig(encoding='utf-8', level=logging.INFO)



def load_data_other_objects(data_origin = 'all'):

	# TODO: Create config file and put paths there
	path_parquet_files = '/home/lmiguel/Data/TDE_classification/AL_data/'
	if data_origin == 'all':
		all_parquet_fnames = glob.glob(os.path.join(path_parquet_files, '*.parquet'))
	else:
		all_parquet_fnames = glob.glob(os.path.join(path_parquet_files,
											  '*{}*.parquet'.format(data_origin)))

	logging.info('Loading parquet files')
	all_obj_df = pd.concat([pd.read_parquet(parquet_fname) for parquet_fname in all_parquet_fnames])

	# Keep only row with all alerts
	all_obj_df['length'] = all_obj_df['cjd'].apply(lambda x: len(x))
	all_obj_df.sort_values(['objectId', 'length'], inplace = True)
	names = np.array(all_obj_df['objectId'])
	mask = np.append((names[:-1] != names[1:]), True)
	all_obj_df = all_obj_df[mask]

	# Rename column
	all_obj_df.rename(columns = {'TNS': 'type',
								  'cdsxmatch': 'type'}, inplace = True)

	logging.info('All files loaded')

	return all_obj_df


def crop_lc_to_rising_time(lc_values, days_to_crop_before = 200):
	"""
	Selects maximum of lightcurve (in any filter), and drops the points after the maximum,
	and before the maxmimum minus "days_to_crop_before" days.

	Parameters
	----------
	lc_values : 2d array
		DESCRIPTION.
	days_to_crop_before : TYPE, optional
		DESCRIPTION. The default is 200.

	Returns
	-------
	lc_values : TYPE
		DESCRIPTION.

	"""
	tmax_idx = np.nanargmax(lc_values[1])
	tmin_idx = np.argmax(lc_values[0] >= lc_values[0][tmax_idx] - days_to_crop_before)
	# Mask from minimum to maximum
	lc_values = lc_values[:, tmin_idx:tmax_idx + 1]

	return lc_values


def plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, ztf_name = ''):
	"""
	Plot lightcurve in all bands, with the rainbow fit.

	Parameters
	----------
	lc_values : 2d array
		Values from the lc (mjd, flux, fluxerr, filter).
	filt_list : list
		list of filters like ['g', 'r', 'i'].
	values_fit : list
		Values from the fit (including chisq).
	err_fit : list
		DESCRIPTION.
	ztf_name : str, optional
		Object ztf name, to put as plot title.


	"""

	mjd, flux, fluxerr, _ = lc_values
	X = np.linspace(lc_values[0].min() - 10, lc_values[0].max() + 10, 500)

	plt.figure(figsize=(12, 8))
	# ax = plt.gca()

	colors = ['green', 'red', 'black']
	for idx, i in enumerate(['g', 'r', 'i']):

		mask = filt_list == i
		f = flux[mask]
		ferr = fluxerr[mask]
		t = mjd[mask]

		rainbow = feature.model(X, i, values_fit)
		plt.errorbar(t, f, yerr=ferr, fmt='o', alpha=.7, color=colors[idx])
		plt.plot(X, rainbow, linewidth=5, label=i, color=colors[idx])
		# Error plots
		generated_params = np.random.multivariate_normal(values_fit[:-1], np.diag(err_fit)**2, 1000)
		generated_lightcurves = np.array([feature.model(X, i, generated_values)
											for generated_values in generated_params])
		generated_envelope = np.nanpercentile(generated_lightcurves, [16, 84], axis=0)
		plt.fill_between(X, generated_envelope[0], generated_envelope[1], alpha=0.2, color=colors[idx])

		plt.title(ztf_name)


def extract_features_for_an_object(row_obj, feature, filt_conv, min_nb_points_fit = 5,
								    show_plots = False):

	name = row_obj.objectId
	trans_type = row_obj.type
	lc_values = np.stack(row_obj[['cjd', 'cmagpsf', 'csigmapsf', 'cfid']])
	# TODO: I could use from here for the other dataframe format if I put
	# lc_values = row_obj[['jd', 'magpsf', 'sigmapsf', 'fid']].values.T
	# row_obj is row_obj = df[df.objectId == 'ZTF18aahqkbt'] within for loop...
	# Remove alerts with nan values
	lc_values = lc_values[:, ~np.isnan(lc_values).any(axis=0)]
	# Flux conversion
	lc_values[1], lc_values[2] = mag2fluxcal_snana(lc_values[1], lc_values[2])
	lc_values = crop_lc_to_rising_time(lc_values)
	# Delete duplicate times
	lc_values = np.delete(lc_values, np.where(np.diff(lc_values[0]) == 0)[0], axis = 1)

	# Convert filter list to g, r, i strings
	filt_list = np.vectorize(filt_conv.get)(lc_values[3].astype(int))
	# Normalization
	norm = lc_values[1][-1]
	lc_values[1], lc_values[2] = lc_values[1] / norm, lc_values[2] / norm

	if lc_values.shape[1] >= min_nb_points_fit:
		values_fit, err_fit = feature(*lc_values[:-1], filt_list)

		if show_plots:
			plot_lightcurve_and_fit(lc_values, filt_list, values_fit, err_fit, ztf_name = name)

		return [name, trans_type] + list(values_fit) + list(err_fit)
	return [name, trans_type] + list(np.full((9), np.nan))


import datetime as dt
start = dt.datetime.now()


# Params
data_origin = 'simbad'

# Initialise variables
csv_file = 'Features_check/features_rainbow_nontdes_{}.csv'.format(data_origin)
band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0}
filt_conv = {1: "g", 2: "r", 3: "i"}
feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline = False)
columns = ['id', 'type', 'reference_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
				'err_reference_time', 'err_amplitude', 'err_rise_time', 'err_temperature']
feature_matrix = pd.DataFrame([], columns = columns)

# Load objects
all_obj_df = load_data_other_objects(data_origin)

# Get features
feature_matrix[columns] = all_obj_df.apply(
	lambda x: extract_features_for_an_object(x, feature, filt_conv, show_plots = False),
											result_type = 'expand', axis = 1)
feature_matrix.dropna(inplace = True)

# Save features
feature_matrix.to_csv(csv_file, index = False)


end = dt.datetime.now()
logging .info("Done in {} seconds.".format(end - start))

import requests
import numpy as np
import glob
import pandas as pd
from io import BytesIO
import sys
import os
import matplotlib.pyplot as plt
import tools_from_sn_classifier as sn_tools
from light_curve.light_curve_py import RainbowFit
import logging
logging.basicConfig(encoding='utf-8', level=logging.INFO)


def get_data_from_FINK(save = True, extended = False):

	# Get object names
	TDEs_hammerstein = pd.read_fwf('ZTF_TDE_Data/Table1_Hammerstein', skiprows = 34, header = None)
	ztf_names = TDEs_hammerstein[1].to_list()

	what_to_retrieve = ['i:jd', 'i:fid', 'i:magpsf', 'i:sigmapsf', 'i:candid', 'i:objectId',
					 'i:ra', 'i:dec']
	if extended:
		extra_cols = ['d:snn_sn_vs_all', 'd:snn_snia_vs_nonia']
		what_to_retrieve = what_to_retrieve + extra_cols

	r = requests.post('https://fink-portal.org/api/v1/objects',
	  json={
		  'objectId': ','.join(ztf_names),
		  'columns': ','.join(what_to_retrieve)
	  }
	)
	fink_df = pd.read_json(BytesIO(r.content))

	fink_df.columns = fink_df.columns.str.lstrip('i:')  # strip prefix

	if save:
		fink_df.to_csv('ZTF_TDE_Data/from_Fink.csv', index = None)

	# tdes_not_in_fink_data = [x for x in ztf_names if x not in fink_df.objectId.unique()]
	return fink_df


def load_forced_photometry_data(fink_df, quality_cuts = True, SNT_thresh = 3):
	"""
	Load forced photometry data, and crossmatch with Fink data positions.
	Conversion of some columns to be like fink data.

	Parameters
	----------
	fink_df : TYPE
		DESCRIPTION.

	Returns
	-------
	all_objects_df : TYPE
		DESCRIPTION.

	"""

	# Load forced-photometry data
	forced_phot_data = glob.glob('ZTF_TDE_Data/forced_photometry/batchfp_*.txt')
	list_of_dfs = []
	for forced_phot_fname in forced_phot_data:
		obj_id = find_objectId_for_fp_data_xmatch_fink(forced_phot_fname, fink_df)
		df_fp = pd.read_csv(forced_phot_fname, comment = '#', sep = ' ')
		df_fp.columns = df_fp.columns.str.strip(',')  # strip prefix
		df_fp['objectId'] = obj_id
		df_fp['fp_fname'] = os.path.basename(forced_phot_fname)
		list_of_dfs.append(df_fp)

	all_objects_df = pd.concat(list_of_dfs)

	all_objects_df.dropna(subset=['objectId'], inplace=True)

	if quality_cuts:
		all_objects_df = all_objects_df[(all_objects_df['infobitssci'] == 0)
		& (all_objects_df['forcediffimflux'] / all_objects_df['forcediffimfluxunc'] > SNT_thresh)]

	return all_objects_df


def merge_features_tdes_SN(csv_tdes, csv_other, out_csv):
	"""
	Merges the new features obtained for TDEs with those from the SN study.

	Parameters
	----------
	csv_tdes : TYPE
		DESCRIPTION.
	csv_other : TYPE
		DESCRIPTION.
	out_csv : TYPE
		DESCRIPTION.

	Returns
	-------
	None.

	"""

	feat_tdes = pd.read_csv(csv_tdes)
	feat_other = pd.read_csv(csv_other)

	merged_df = pd.concat([feat_tdes, feat_other])

	merged_df.to_csv(out_csv, index = False)


def crop_lc_to_rising_part(converted_df: pd.DataFrame, minimum_nb_obs: int = 4, days_to_crop = 200,
						   nb_points_after_max = 2, save_csv = True, drop_duplicates = True):
	"""
	Crop the light-curve to retain only the rising part. Drop every observation after the max flux.
	Keep observations of an object only if the object presents at least "minimum_nb_obs" observations.
	We crop the beginning to be 100 days before the maximum flux value.
	# TODO: Maybe trim based on histogram (get e.g. 90 pr cent)

	Parameters
	----------
	converted_df : pd.DataFrame
		Dataset with columns ['objectId', 'type', 'MJD', 'FLT','FLUXCAL', 'FLUXCALERR'].
		Each row is an alert.
	minimum_nb_obs : int, optional
		Minimum number of observations (otherwise drop alerts of object). The default is 3.
	save_csv : bool, optional
		Wehther to save the output dataframe onto a csv. The default is True.

	Returns
	-------
	converted_df_early :  pd.DataFrame
		Data prepared for the fitting, after dropping the decaying part of the LC.

	"""

	df_list = []
	for indx in range(np.unique(converted_df['id'].values).shape[0]):

		name = np.unique(converted_df['id'].values)[indx]
		obj_flag = converted_df['id'].values == name
		obj_df = converted_df[obj_flag]
		tmax = 0
		for filt in ['g', 'r']:
			filt_df = obj_df[obj_df['FLT'] == filt].copy()
			if len(filt_df) > 0:
				tmax = max(tmax, filt_df['MJD'][filt_df['FLUXCAL'].idxmax()])

		tmin = tmax - days_to_crop
		obj_df = obj_df[(obj_df.MJD <= tmax) & (obj_df.MJD >= tmin)]
		if len(obj_df) > minimum_nb_obs:
			df_list.append(obj_df)

	converted_df_early = pd.concat(df_list)
	if drop_duplicates:
		converted_df_early.drop_duplicates(inplace = True)
	if save_csv:
		converted_df_early.to_csv('input_for_feature_extractor.csv', index = False)
	return converted_df_early


def is_unique(s):
	""" Check if all values of a (pandas) series are equal. """
	a = s.to_numpy()
	return (a[0] == a).all()


def find_objectId_for_fp_data_xmatch_fink(forced_phot_fname, df_fink, deg_tolerance = 0.001):
	"""
	Correlate the forced photometry data with the Fink data, to find the object ID corresponding
	to the forced-photometry data file.

	Parameters
	----------
	forced_phot_fname : str
		filename containing the forced_phot data of one object.
	df_fink : pd.DataFrame
		Data from Fink with all the objects at interest.
	deg_tolerance : float, optional
		Margin in degrees, for the search in RA and DEC. The default is 0.001.

	Returns
	-------
	obj_id : str
		Object Identifier.

	"""

	with open(forced_phot_fname) as f:
		for i, line in enumerate(f):
			if i == 3:
				req_ra = float(line.split(' ')[-2])
			elif i ==4:
				req_dec = float(line.split(' ')[-2])
			elif i > 4:
				break

	# TODO: do it with astropy
	df_obj = df_fink[(df_fink.ra > req_ra - deg_tolerance) & (df_fink.ra < req_ra + deg_tolerance) &
			 (df_fink.dec > req_dec - deg_tolerance) & (df_fink.dec < req_dec + deg_tolerance)]
	if len(df_obj) == 0:
		print('Error while correlating the forced-phometry data with the objectId: '+
				'No object was found in this position, for ' + forced_phot_fname)

		obj_id = os.path.basename(forced_phot_fname)

	elif is_unique(df_obj.objectId):
		obj_id = df_obj.objectId.iloc[0]

	else:
		print('Error while correlating the forced-phometry data with the objectId: '+
				'more than one object are within the given position given.')
		obj_id = None
	return obj_id


def find_objectId_for_fp_data_from_fink_resolver(forced_phot_fname):
	"""
	Find the object ID corresponding to the forced-photometry data file, using the Fink resolver
	with position.

	Parameters
	----------
	forced_phot_fname : str
		filename containing the forced_phot data of one object.

	Returns
	-------
	obj_id : str
		Object Identifier.

	"""
	with open(forced_phot_fname) as f:
		for i, line in enumerate(f):
			if i == 3:
				req_ra = float(line.split(' ')[-2])
			elif i ==4:
				req_dec = float(line.split(' ')[-2])
			elif i > 4:
				break

	FINK_API_URL = 'https://fink-portal.org'

	# Call fink conesearch
	r = requests.post('{}/api/v1/explorer'.format(FINK_API_URL),
				   json={'ra': req_ra, 'dec': req_dec, 'radius': 5.0, 'columns': 'i:objectId'})

	if r.json() == []:
	    logging.warning('Unable to cross-correlate object from file ' + forced_phot_fname
					 + 'with Fink data. No counterpart in Fink.')
	    return forced_phot_fname

	objectId = r.json()[0]['i:objectId']

	r = requests.post('{}/api/v1/resolver'.format(FINK_API_URL),
				   json={'resolver': 'tns', 'name': objectId, 'reverse': True})
	return r.json()


def convert_forced_phot_df(df):

	df.rename(columns = { 'objectId' : 'id',
					  'jd': 'MJD'}, inplace = True)
	df['magpsf'] = df['zpdiff'] - 2.5 * np.log10(df['forcediffimflux'])
	df['FLUXCAL'] = 10 ** (-0.4 * df['magpsf']) * 10 ** (11)

	df['sigmapsf'] = 1.0857 * df['forcediffimfluxunc'] / df['forcediffimflux']
	df['FLUXCALERR'] =  9.21034 * 10 ** 10 * np.exp(-0.921034 * df['magpsf']) * df['sigmapsf']


	df['FLT']  = df['filter'].str[-1]
	df['type'] = 'TDE'
	df.reset_index(inplace = True)

	return df[['id', 'type', 'MJD', 'FLT', 'FLUXCAL', 'FLUXCALERR', 'fp_fname']]


def convert_df(fink_df, data_origin = 'fink'):


	if data_origin =='forced_phot':
		df = load_forced_photometry_data(fink_df)
		converted_df = convert_forced_phot_df(df)
	elif data_origin == 'fink':
		converted_df = sn_tools.convert_full_dataset(fink_df, obj_id_header='objectId')
	else:
		print('wrong string given')
		sys.exit()
	return converted_df


def extract_rainbow_feat(df_to_extract, show_plots = True):
	"""
	Preliminary function to extract rainbow features on the Dataframe.

	Parameters
	----------
	df_to_extract : TYPE
		DESCRIPTION.
	show_plots : TYPE, optional
		DESCRIPTION. The default is True.

	Returns
	-------
	feature_matrix : TYPE
		DESCRIPTION.

	"""

	columns = ['id', 'type', 'reference_time', 'amplitude', 'rise_time', 'temperature', 'r_chisq',
					'err_reference_time', 'err_amplitude', 'err_rise_time', 'err_temperature']
	# Effective wavelengths in Angstrom
	band_wave_aa = {"g": 4770.0, "r": 6231.0, "i": 7625.0}
	features_all = []
	for indx in range(np.unique(df_to_extract['id'].values).shape[0]):
		# print('\r Objects processed: {}'.format(indx + 1), end='\r')
		print('Objects processed: {}'.format(indx + 1))

		ztf_name = np.unique(df_to_extract['id'].values)[indx]

		obj_flag = df_to_extract['id'].values == ztf_name
		obj_df = df_to_extract[obj_flag]
		obj_df['MJD'] = np.where(obj_df['MJD'].duplicated(keep=False), obj_df['MJD'] +
						obj_df.groupby('MJD').cumcount().add(0.25).astype(float), obj_df['MJD'])
		if len(obj_df) > 4:

			transient_type = df_to_extract[obj_flag].iloc[0]['type']
			print(ztf_name)

			obj_df.sort_values('MJD', inplace = True)

			line = [ztf_name, transient_type]

			flux = obj_df['FLUXCAL']
			fluxerr = obj_df['FLUXCALERR']
			band = obj_df['FLT']
			norm = flux.max()
			flux, fluxerr = flux / norm, fluxerr / norm
			mjd = obj_df['MJD']
			feature = RainbowFit.from_angstrom(band_wave_aa, with_baseline = False)
			values, errors = feature(mjd, flux, sigma=fluxerr, band=band)
			r_chisq = values[-1:]
			features = values[:-1]
			line.extend((list(features) + list(r_chisq) + list(errors)))


			features_all.append(line)

			if show_plots:
				X = np.linspace(mjd.min(), mjd.max() + 20, 500)
				if 'fp_fname' in df_to_extract.columns:
					fp_fname = df_to_extract[obj_flag].fp_fname.iloc[0]
				else:
					fp_fname = ' '

				plt.figure(figsize=(12, 8))
				ax = plt.gca()

				colors = ['green', 'red', 'black']
				for idx, i in enumerate(["g", "r", "i"]):
					mask = band == i
					f = flux[mask]
					ferr = fluxerr[mask]
					t = mjd[mask]
					rainbow = feature.model(X, i, values)
					plt.errorbar(t, f, yerr=ferr, fmt='o', alpha=.7, color=colors[idx])
					plt.plot(X, rainbow, linewidth=5, label=i, color=colors[idx])
					# Error plots
					generated_parameters = np.random.multivariate_normal(features, np.diag(errors)**2,1000)
					generated_lightcurves = np.array([feature.model(X, i, generated_values) for generated_values in generated_parameters])
					generated_envelope = np.nanpercentile(generated_lightcurves, [16, 84], axis=0)
					plt.fill_between(X, generated_envelope[0], generated_envelope[1],alpha=0.2,color=colors[idx])

					plt.title('{}, {}'.format(ztf_name, fp_fname))
					# ax.text(0.5, 0.5, '{}'.format(features[-1]), transform=ax.transAxes)


				plt.legend()

	feature_matrix = pd.DataFrame(features_all, columns=columns)
	return feature_matrix


def generate_features_tdes(data_origin = 'forced_phot', feat_extractor = 'rainbow',
						   overwrite_fink_df = False, show_plots = False):

	if data_origin == 'fink_extended':
		converted_df = pd.read_csv('ZTF_TDE_Data/all_tde_in_ztf.csv', dtype={'id': str})
		converted_df['type'] = 'TDE'
		converted_df.drop(columns = 'id', inplace=True)
		converted_df.rename(columns = {'objectId': 'id'}, inplace = True)
	else:
		if overwrite_fink_df:
			fink_df,_ = get_data_from_FINK(save = True, extended = True)
		else:
			fink_df = pd.read_csv('ZTF_TDE_Data/from_Fink.csv')

		converted_df = convert_df(fink_df, data_origin)

	converted_df_early = crop_lc_to_rising_part(converted_df, save_csv = True)


# 	# Obtain features and save
	if feat_extractor == 'rainbow':
		feature_matrix = extract_rainbow_feat(converted_df_early, show_plots = show_plots)
		feature_matrix.to_csv('Features_check/features_rainbow_tdes.csv', index = None)

	else:
		feature_matrix = sn_tools.featurize_full_dataset(converted_df_early, screen = True)
		feature_matrix.to_csv('Features_check/features_tdes.csv', index = None)
		merge_features_tdes_SN('Features_check/features_tdes.csv', 'Features_check/features_non_tdes.csv',
							   'Features_check/merged_features.csv')


if __name__ == '__main__':

	data_origin = 'forced_phot'
	data_origin = 'fink_extended'
	feat_extractor = 'rainbow'

	# Get features TDEs
	generate_features_tdes(data_origin, feat_extractor, overwrite_fink_df = False, show_plots = True)

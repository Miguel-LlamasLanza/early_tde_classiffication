#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 15:55:26 2023

@author: lmiguel

Script aimed at doing a quick analysis the classification provided by Fink for the 30 TDEs in our sample.
"""

import pandas as pd
import matplotlib.pyplot as plt

# Import data
fink_df = pd.read_csv('ZTF_TDE_Data/from_Fink.csv')

max_sn_class_list = []
last_sn_class_list = []
nb_alerts_list = []
for objectId in fink_df.objectId.unique():

	df_obj = fink_df[fink_df.objectId == objectId]
	df_obj = df_obj.sort_values('jd').reset_index()
	nb_alerts = len(df_obj)

	nb_alerts_list.append(nb_alerts)

	# Get the maximum classification score for the object
	max_class_SN = df_obj['d:snn_sn_vs_all'].max()
	max_sn_class_list.append(max_class_SN)

	# Get last classification score
	last_sn_class = df_obj['d:snn_sn_vs_all'].iloc[-1]
	last_sn_class_list.append(last_sn_class)
	if nb_alerts > 10:

		plt.figure()
		ax = plt.gca()
		df_obj.plot(x = 'jd', y = 'd:snn_sn_vs_all', ax = ax, label = 'Max Score', c = 'black')
		ax.set_xlabel('Time (days)')

		ax_new = ax.twinx()
		for filt, filt_str in zip([1, 2], ['g', 'r']):
			df_obj_filt = df_obj[df_obj.fid == filt]
			ax_new.errorbar(df_obj_filt.jd, df_obj_filt.magpsf, yerr = df_obj_filt.sigmapsf,
					  label = 'LC (flux), FLT: ' + filt_str, fmt= '+')
		ax_new.set_ylabel('MAgnitude')
		ax.legend(loc='upper left')
		ax_new.legend(loc='upper right')

plt.figure()
plt.hist(max_sn_class_list, histtype = 'step', bins = 15)
plt.xlabel('Maximum SN classification score')
plt.ylabel('Number of objects')
plt.text(0.5, 10, 'Total number: {}'.format(len(fink_df.objectId.unique())))

plt.figure()
plt.hist(nb_alerts_list, histtype = 'step', bins = 15)
plt.xlabel('Number of alerts')
plt.ylabel('Number of objects')
plt.text(0.5, 10, 'Total number: {}'.format(len(fink_df.objectId.unique())))


plt.figure()
plt.scatter(nb_alerts_list, max_sn_class_list, s = 15, label = 'Max class')
plt.scatter(nb_alerts_list, last_sn_class_list, s = 15, label ='Last class', marker = '+')

plt.xlabel('Number of alerts per object')
plt.ylabel('Maximum SN classification score')

plt.legend()

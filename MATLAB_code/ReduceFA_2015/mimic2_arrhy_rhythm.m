% Extract Critical arrhythmia rhythm from MIMIC II ver2 db
% Created by HeeHwan Park

% Data from 'Aboukhalil, Anton, et al. "Reducing false alarm rates for
% critical arrhythmias using the arterial blood pressure waveform." Journal of biomedical informatics 41.3 (2008): 442-451.'

clc;
clear;

dbfolder = '/media/salab-heehwan/HDD_1TB/WFDB_data/MIMIC2_ver2/';
filenum = 'a40017';

% alMa: True alarm
% annotations: All alarm
% samples: record

wavefile = fopen(strcat(dbfolder,'samples',filenum,'.csv'));
waverecord = textscan(wavefile,'%d %s', -inf, 'Delimiter', ',', 'EmptyValue', -Inf, 'HeaderLines', 2);
fclose(wavefile);

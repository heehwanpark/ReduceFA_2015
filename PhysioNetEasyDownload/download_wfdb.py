# Download wfdb files in www.physioNet.org
import os
import urllib
import urllib2
import time

def loadBasicInfo(url_folder, path, data_num):
    # Header
    urllib.urlretrieve(url_folder+'/'+data_num+'.hea', path+'/'+data_num+'.hea')
    # Manual annotation
    urllib.urlretrieve(url_folder+'/'+data_num+'.alM', path+'/'+data_num+'.alM')
    # Alarms by machine
    urllib.urlretrieve(url_folder+'/'+data_num+'.alarms', path+'/'+data_num+'.alarms')
    # Layout
    urllib.urlretrieve(url_folder+'/'+data_num+'_layout.hea', path+'/'+data_num+'_layout.hea')
    print '...basic information of ' + data_num + ' are saved successfully'

def loadSubData(url_folder, path, sub_dat, sub_hea):
    urllib.urlretrieve(url_folder+'/'+sub_dat, path+'/'+sub_dat)
    urllib.urlretrieve(url_folder+'/'+sub_hea, path+'/'+sub_hea)

list_file = open('mimic2_ver2_annotation_list', 'r')
for data_num in list_file:
    data_num = data_num.strip()
    path = '/media/heehwan/308446B784467EFA/WFDB_data/'+data_num
    os.mkdir(path)

    print '...downloading ' + data_num + ' folder'

    url_folder = 'http://www.physionet.org/physiobank/database/mimic2db/' + data_num

    try:
        loadBasicInfo(url_folder, path, data_num)
    except IOError as e:
        print e
        time.sleep(5)
        loadBasicInfo(url_folder, path, data_num)

    # Read header file
    with open(path+'/'+data_num+'.hea') as hf:
        count = 1
        for line in hf:
            if count > 2 and not line.startswith('~'):
                string_list = line.split(' ')
                subdata_num = string_list[0]
                print subdata_num

                sub_dat = subdata_num + '.dat'
                sub_hea = subdata_num + '.hea'
                try:
                    loadSubData(url_folder, path, sub_dat, sub_hea)
                except IOError as e:
                    print e
                    time.sleep(5)
                    loadSubData(url_folder, path, sub_dat, sub_hea)
            count += 1
        hf.close()

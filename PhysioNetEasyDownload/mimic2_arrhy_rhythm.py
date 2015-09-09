import csv
import numpy
import datetime
import h5py

def extractWaveFromMimicfile(dbfolder, filenum, h5file_obj):
    print("Start extracting from File " + filenum)

    wave = []
    ann_list = []
    alM_list = []

    # open and read csv (sample) File
    with open(dbfolder+'samples'+filenum+'.csv', 'rb') as csvfile:
        csvreader = csv.reader(csvfile)
        # skip header
        for i in xrange(2):
            csvreader.next()
        # read csv
        for row in csvreader:
            try:
                value = float(row[1])
            except Exception as e:
                value = 0
            wave.append(value)

    # open and read annotation (machine alarm) file
    with open(dbfolder+'annotations'+filenum+'.txt', 'rb') as annfile :
        annfile.next()
        for line in annfile :
            line_list = line.split()
            if ('ASYSTOLE' in line_list or
                '**BRADY' in line_list or
                '***BRADY' in line_list or
                '**TACHY' in line_list or
                '***TACHY' in line_list or
                'V-FIB/TACH' in line_list or
                'V-TACH' in line_list) :
                ann_list.append(float(line_list[2]))

    # open and read alM (annotated by human experts) file
    with open(dbfolder+'alM'+filenum+'.txt', 'rb') as alMfile :
        alMfile.next()
        for line in alMfile:
            line_list = line.split()
            alM_list.append(float(line_list[2]))

    ann_num = len(ann_list)
    alM_num = len(alM_list)
    print("Number of entire alarms in " + filenum + ": " + str(ann_num))
    print("Number of true alarms in " + filenum + ": " + str(alM_num))

    wave_length = len(wave)
    hz = 125

    inputs = []
    targets = []

    # for check
    tar_1 = 0
    tar_0 = 0

    start_idx = 0
    end_idx = start_idx + 10*hz
    while end_idx < len(wave):
        for ann in ann_list:
            if ann >= start_idx and ann <= end_idx:
                input = wave[start_idx:end_idx]
                for i in xrange(10*hz):
                    if numpy.absolute(input[i]) > 3:
                        input[i] = 0
                mean = numpy.mean(input)
                std = numpy.std(input)
                if std == 0:
                    input = input - mean
                else:
                    input = (input - mean)/std

                if ann in alM_list :
                    target = 1
                    tar_1 = tar_1 + 1
                else :
                    target = 0
                    tar_0 = tar_0 + 1

                inputs.append(input)
                targets.append(target)

        start_idx = start_idx + 1*hz
        end_idx = start_idx + 10*hz
    print(tar_1)
    print(tar_0)

    # save data
    print("Saving data ...")

    inputs_array = numpy.asarray(inputs)
    targets_array = numpy.asarray(targets)

    print(inputs_array.shape)
    print(targets_array.shape)

    grp = h5file_obj.create_group(filenum)
    grp.create_dataset("inputs", data=inputs_array)
    grp.create_dataset("targets", data=targets_array)

if __name__ == "__main__":
    dbfolder = '/media/heehwan/HDD_1TB/WFDB_data/MIMIC2_ver2/'
    logfile = open('error_log', 'w')
    f = h5py.File("mimic2_savefile.h5", "w")
    with open('mimic2_annotation_list_v1', 'r') as filelist:
        for filenum in filelist:
            filenum = filenum.strip()
            try:
                extractWaveFromMimicfile(dbfolder, filenum, f)
            except Exception as e:
                logfile.write(filenum + '\n')
                logfile.write(e + '\n')
    f.close()
    logfile.close()

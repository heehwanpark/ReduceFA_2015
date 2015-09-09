import csv
import numpy
import datetime
import h5py

def checkNumAlarm(dbfolder, filenum, ann_num, alM_num):
    ann_list = []
    alM_list = []

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

    ann_num = ann_num + len(ann_list)
    alM_num = alM_num + len(alM_list)

    return ann_num, alM_num

def extractWaveFromMimicfile(dbfolder, filenum, h5file_obj):
    print("Start extracting from File " + filenum)

    wave = []
    alM_list = []
    chan_list = []

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

    # # open and read annotation (machine alarm) file
    # with open(dbfolder+'annotations'+filenum+'.txt', 'rb') as annfile :
    #     annfile.next()
    #     for line in annfile :
    #         line_list = line.split()
    #         if ('ASYSTOLE' in line_list or
    #             '**BRADY' in line_list or
    #             '***BRADY' in line_list or
    #             '**TACHY' in line_list or
    #             '***TACHY' in line_list or
    #             'V-FIB/TACH' in line_list or
    #             'V-TACH' in line_list) :
    #             ann_list.append(float(line_list[2]))

    # open and read alM (annotated by human experts) file
    with open(dbfolder+'alM'+filenum+'.txt', 'rb') as alMfile :
        alMfile.next()
        for line in alMfile:
            line_list = line.split()
            alM_list.append(float(line_list[2]))
            chan_list.append(float(line_list[5]))

    alM_num = len(alM_list)
    print("Number of entire alarms in " + filenum + ": " + str(alM_num))

    wave_length = len(wave)
    hz = 125

    inputs = []
    targets = []

    start_idx = 0
    end_idx = start_idx + 10*hz
    while end_idx < len(wave):
        for i in xrange(len(alM_list)):
            if alM_list[i] >= start_idx and alM_list[i] <= end_idx:
                input = wave[start_idx:end_idx]
                for j in xrange(10*hz):
                    if numpy.absolute(input[j]) > 3:
                        input[j] = 0
                mean = numpy.mean(input)
                std = numpy.std(input)
                if std == 0:
                    input = input - mean
                else:
                    input = (input - mean)/std

                if chan_list[i] == 1:
                    target = 1
                elif chan_list[i] == 3:
                    target = 0
                else:
                    print("DB error!")
                    break

                inputs.append(input)
                targets.append(target)

        start_idx = start_idx + 1*hz
        end_idx = start_idx + 10*hz

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
    # ann_num = 0
    # alM_num = 0
    # with open('mimic2_annotation_list_v1', 'r') as filelist:
    #     for filenum in filelist:
    #         filenum = filenum.strip()
    #         ann_num, alM_num = checkNumAlarm(dbfolder, filenum, ann_num, alM_num)
    # print(ann_num)
    # print(alM_num)

import csv
import numpy
import datetime
import h5py

def extractWaveFromMimicfile(dbfolder, filenum, h5file_obj, all_alarms, true_alarms):
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

    # open and read alM (annotated by human experts) file
    with open(dbfolder+'alM'+filenum+'.txt', 'rb') as alMfile :
        alMfile.next()
        for line in alMfile:
            line_list = line.split()
            alM_list.append(float(line_list[2]))
            chan_list.append(float(line_list[5]))

            all_alarms = all_alarms + 1
            if int(line_list[5]) == 1:
                true_alarms = true_alarms + 1

    alM_num = len(alM_list)
    print("Number of entire alarms in " + filenum + ": " + str(alM_num))

    wave_length = len(wave)
    hz = 125

    inputs = []
    targets = []

    for i in xrange(alM_num) :
        segment_end = int(alM_list[i])
        segment_start = segment_end - 10*hz
        if segment_start >= 0 and segment_end < wave_length :
            input = wave[segment_start:segment_end]
            mean = numpy.mean(input)
            std = numpy.std(input)
            if std == 0:
                input = input - mean
            else:
                input = (input - mean)/std

            chan_num = int(chan_list[i])
            if chan_num == 1:
                target = 1
            elif chan_num == 3:
                target = 0
            else:
                print("DB error!")
                break

            inputs.append(input)
            targets.append(target)

    # save data
    print("Saving data ...")

    inputs_array = numpy.asarray(inputs)
    targets_array = numpy.asarray(targets)

    print(inputs_array.shape)
    print(targets_array.shape)

    # grp = h5file_obj.create_group(filenum)
    groupname = '/'+filenum
    grp = h5file_obj[groupname]
    grp.create_dataset("inputs", data=inputs_array)
    grp.create_dataset("targets", data=targets_array)

    return all_alarms, true_alarms

if __name__ == "__main__":
    dbfolder = '/media/heehwan/HDD_1TB/WFDB_data/MIMIC2_ver2/'
    logfile = open('error_log', 'w')
    f = h5py.File("/home/heehwan/Workspace/Data/ReduceFA_2015/mimic2_savefile_v2.h5", "a")
    # f = h5py.File("mimic2_savefile_v2.h5", "w")

    all_alarms = 0
    true_alarms = 0

    filenum = "a40454"
    all_alarms, true_alarms = extractWaveFromMimicfile(dbfolder, filenum, f, all_alarms, true_alarms)
    print(str(all_alarms) + " : " + str(true_alarms))

    # with open('mimic2_annotation_list_v1', 'r') as filelist:
    #     for filenum in filelist:
    #         filenum = filenum.strip()
    #         try:
    #             all_alarms, true_alarms = extractWaveFromMimicfile(dbfolder, filenum, f, all_alarms, true_alarms)
    #             print(str(all_alarms) + " : " + str(true_alarms))
    #         except Exception as e:
    #             print(e)
    #             logfile.write(filenum + '\n')
    #
    # f.close()
    # logfile.close()

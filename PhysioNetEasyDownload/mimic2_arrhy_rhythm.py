import csv
import numpy
import datetime
import h5py

def makeMimicDB():
    dbfolder = '/media/salab-heehwan/HDD_1TB/WFDB_data/MIMIC2_ver2/'
    with open('mimic2_annotation_list_v1', 'r') as filelist :
        inputs = []
        targets = []

        ann_num = 0
        alM_num = 0

        for filenum in filelist:
            filenum = filenum.strip()
            print("File " + filenum)

            wave = []
            ann_list = []
            alM_list = []

            with open(dbfolder+'samples'+filenum+'.csv', 'rb') as csvfile :
                csvreader = csv.reader(csvfile)
                # skip header
                for i in xrange(2) :
                    csvreader.next()
                # read csv
                for row in csvreader:
                    try:
                        value = float(row[1])
                    except Exception as e:
                        value = 0
                    wave.append(value)

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

            with open(dbfolder+'alM'+filenum+'.txt', 'rb') as alMfile :
                alMfile.next()
                for line in alMfile:
                    line_list = line.split()
                    alM_list.append(float(line_list[2]))

            ann_num = ann_num + len(ann_list)
            alM_num = alM_num + len(alM_list)

            print(ann_num)
            print(alM_num)

            mean = numpy.mean(wave)
            std = numpy.std(wave)
            wave = (wave - mean)/std

            # delete weird value
            for i in xrange(len(wave)) :
                if numpy.absolute(wave[i]) > 10*std :
                    wave[i] = 0

            wave_length = len(wave)
            hz = 125

            start_idx = 0
            end_idx = start_idx + 10*hz
            while end_idx < len(wave) :
                for ann in ann_list :
                    if ann >= start_idx and ann <= end_idx :
                        input = wave[start_idx:end_idx]
                        if ann in alM_list :
                            target = 1
                        else :
                            target = 0
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

    with h5py.File("samplefile.h5", "w") as f :
        f.create_dataset("inputs", data=inputs_array)
        f.create_dataset("targets", data=targets_array)

if __name__ == "__main__":
    makeMimicDB()

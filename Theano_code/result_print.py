import pickle
import csv

result = pickle.load(open('result_400_3_v1.pkl', 'rb'))
with open('best_result_v1.csv', 'wb') as csvfile1:
    csvwriter = csv.writer(csvfile1, delimiter=',')
    for j in xrange(10):
        ele_i = result[j]
        csvwriter.writerow([ele_i[0], ele_i[1], ele_i[2], ele_i[3]])

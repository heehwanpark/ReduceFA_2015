import pickle
import csv

cond_list = pickle.load(open('condition_list.pkl', 'rb'))
cond_result_list = pickle.load(open('condition_result.pkl', 'rb'))

f_best = 0
f_score_ge_40 = []

for i in xrange(len(cond_result_list)):
    cur_result = cond_result_list[i]
    f_score = cur_result[2]
    if f_score > f_best:
        f_best = f_score
        best_idx = i
    if f_score > 0.40:
        f_score_ge_40.append(i)

best_cond = cond_list[best_idx]
best_result = cond_result_list[best_idx]

print 'Best condition: '
print '- hidden architecture', best_cond[0]
print '- regularization rate', best_cond[1]
print (('- learning rate %f, and batch size %i') % (best_cond[2], best_cond[3]))

print 'Best result: '
print (('- Mean training accuracy %f %%, testing accuracy %f %%, f-score %f') % (best_result[0], best_result[1], best_result[2]))
print (('- Std of training accuracy %f %%, testing accuracy %f %%, f-score %f') % (best_result[3], best_result[4], best_result[5]))

print '#####################################'
print '... exporting good results'

with open('good_conditions.csv', 'wb') as csvfile1:
    csvwriter = csv.writer(csvfile1, delimiter=',')
    csvwriter.writerow(['Hidden architecture', 'regularization rate', 'Learning rate', 'Batch size',
                        'mean train', 'std train', 'mean test', 'std test', 'mean f-score', 'std f-score'])
    for j in xrange(len(f_score_ge_40)):
        ele_i = f_score_ge_40[j]
        h_architecture = cond_list[ele_i][0]
        reg_rate = cond_list[ele_i][1]
        str_arch = '-'.join(str(e) for e in h_architecture)
        str_reg = '-'.join(str(rr) for rr in reg_rate)
        csvwriter.writerow([str_arch, str_reg, cond_list[ele_i][2], cond_list[ele_i][3],
                            cond_result_list[ele_i][0], cond_result_list[ele_i][3],
                            cond_result_list[ele_i][1], cond_result_list[ele_i][4],
                            cond_result_list[ele_i][2], cond_result_list[ele_i][5]])

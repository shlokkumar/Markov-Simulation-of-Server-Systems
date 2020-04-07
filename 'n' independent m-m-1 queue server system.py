import numpy as np
from numpy import array
import random as ran
import matplotlib.pyplot as plt

lam = [7, 8, 9, 9.5]
mu = 1
N = 50090
m = 10
total_iter = 10

super_average_1 = []
for la in range(len(lam)):

  for_each_lambda_departure = [0 for i in range(N+20)]

  for k in range(total_iter):

    l = lam[la]
    service_times = []
    arrival_times = []
    start_times_customers = []
    end_time_customers = []

    all_queues_q_and_s = []

    service_times.append(-1*np.log(1 - np.random.uniform())/mu)
    arrival_times.append(-1*np.log(1 - np.random.uniform())/l)

    for i in range(1, N):
        service_times.append(-1*np.log(1 - np.random.uniform())/mu)
        arrival_times.append(arrival_times[i-1] + (-1*np.log(1 - np.random.uniform())/l))

    start_time_server = [0 for i in range(m)]
    end_time_server = [0 for i in range(m)]
    start_time_customer = [0 for i in range(N)]
    end_time_customer = [0 for i in range(N)]

    for i in range(N):

      if arrival_times[i] >= end_time_customer[5000] and end_time_customer[5000] != 0:
        break

      if i == 0:
        n = ran.randint(0, m-1)
        queue = []
        server = []
        server.append(i)
        current_arrival = [None] * m
        current_arrival[n] = [server, queue]
        all_queues_q_and_s.append(current_arrival)

        start_time_server[n] = start_time_server[n] + arrival_times[i]
        start_time_customer[i] = arrival_times[i]
        end_time_server[n] = end_time_server[n] + arrival_times[i] + service_times[i]
        end_time_customer[i] = start_time_customer[i] + service_times[i]
        

      else:
        prev_status = list(all_queues_q_and_s[i-1])
        
        for s in range(len(prev_status)):
          if prev_status[s] != None:
            se, qu = prev_status[s]

            if arrival_times[i] < start_time_customer[se[0]] or arrival_times[i] > end_time_customer[se[0]]:
              flag = -1
              for j in range(len(qu)):
                if arrival_times[i] >= start_time_customer[qu[j]] and arrival_times[i] <= end_time_customer[qu[j]]:
                  flag = j
                  break
              
              if flag == -1:
                prev_status[s] = None
              
              else:
                se[0] = qu[flag]
                if flag == len(qu) - 1:
                  qu = []
                else:
                  qu = qu[flag+1:]
                prev_status[s] = [se, qu]

        all_queues_q_and_s.append(prev_status)

        n1 = ran.randint(0, m-1)
        n2 = ran.randint(0, m-1)

        while n1 == n2:
          n1 = ran.randint(0, m-1)
          n2 = ran.randint(0, m-1) 		

        if all_queues_q_and_s[i][n1] != None and all_queues_q_and_s[i][n2] == None:
          se = []
          qu = []
          se.append(i)
          all_queues_q_and_s[i][n2] = [se, qu]

          start_time_server[n2] = arrival_times[i]
          end_time_server[n2] = arrival_times[i] + service_times[i]

          start_time_customer[i] = start_time_server[n2]
          end_time_customer[i] = start_time_customer[i] + service_times[i]

        elif all_queues_q_and_s[i][n1] == None and all_queues_q_and_s[i][n2] != None:
          se = []
          qu = []
          se.append(i)
          all_queues_q_and_s[i][n1] = [se, qu]

          start_time_server[n1] = arrival_times[i]
          end_time_server[n1] = arrival_times[i] + service_times[i]

          start_time_customer[i] = start_time_server[n1]
          end_time_customer[i] = start_time_customer[i] + service_times[i]

        elif all_queues_q_and_s[i][n1] == None and all_queues_q_and_s[i][n2] == None:
          se = []
          qu = []
          se.append(i)
          all_queues_q_and_s[i][n1] = [se, qu]

          start_time_server[n1] = arrival_times[i]
          end_time_server[n1] = arrival_times[i] + service_times[i]
          start_time_customer[i] = start_time_server[n1]
          end_time_customer[i] = start_time_customer[i] + service_times[i]

        else:
          se1, qu1 = all_queues_q_and_s[i][n1]
          se2, qu2 = all_queues_q_and_s[i][n2]

          l1 = len(se1) + len(qu1)
          l2 = len(se2) + len(qu2)

          if l1 <= l2:
            qu1.append(i)
            all_queues_q_and_s[i][n1] = [se1, qu1]

            start_time_customer[i] = end_time_customer[qu1[len(qu1)-2]]
            end_time_customer[i] = start_time_customer[i] + service_times[i]

          else:
            qu2.append(i)
            all_queues_q_and_s[i][n2] = [se2, qu2]
            start_time_customer[i] = end_time_customer[qu2[len(qu2)-2]]
            end_time_customer[i] = start_time_customer[i] + service_times[i]

      
    req_customer = len(all_queues_q_and_s)
    wait_time_customer = [0 for i in range(req_customer)]
    for i in range(req_customer):
      wait_time_customer[i] = abs(end_time_customer[i] - arrival_times[i])

    for idx in range(req_customer):
      for_each_lambda_departure[idx] = for_each_lambda_departure[idx] + wait_time_customer[idx]

  for q in range(N+20):
    for_each_lambda_departure[q] = for_each_lambda_departure[q] / total_iter
  
  super_average_1.append(for_each_lambda_departure)

super_average_2 = []
for la in range(len(lam)):

  for_each_lambda_departure = [0 for i in range(N+20)]

  for k in range(total_iter):

    l = lam[la]
    service_times = []
    arrival_times = []
    start_times_customers = []
    end_time_customers = []

    all_queues_q_and_s = []

    service_times.append(-1*np.log(1 - np.random.uniform())/mu)
    arrival_times.append(-1*np.log(1 - np.random.uniform())/l)

    for i in range(1, N):
        service_times.append(-1*np.log(1 - np.random.uniform())/mu)
        arrival_times.append(arrival_times[i-1] + (-1*np.log(1 - np.random.uniform())/l))

    start_time_server = [0 for i in range(m)]
    end_time_server = [0 for i in range(m)]
    start_time_customer = [0 for i in range(N)]
    end_time_customer = [0 for i in range(N)]

    for i in range(N):

      if arrival_times[i] >= end_time_customer[50000] and end_time_customer[50000] != 0:
        break

      if i == 0:
        n = ran.randint(0, m-1)
        queue = []
        server = []
        server.append(i)
        current_arrival = [None] * m
        current_arrival[n] = [server, queue]
        all_queues_q_and_s.append(current_arrival)

        start_time_server[n] = start_time_server[n] + arrival_times[i]
        start_time_customer[i] = arrival_times[i]
        end_time_server[n] = end_time_server[n] + arrival_times[i] + service_times[i]
        end_time_customer[i] = start_time_customer[i] + service_times[i]
        

      else:
        prev_status = list(all_queues_q_and_s[i-1])
        
        for s in range(len(prev_status)):
          if prev_status[s] != None:
            se, qu = prev_status[s]

            if arrival_times[i] < start_time_customer[se[0]] or arrival_times[i] > end_time_customer[se[0]]:
              flag = -1
              for j in range(len(qu)):
                if arrival_times[i] >= start_time_customer[qu[j]] and arrival_times[i] <= end_time_customer[qu[j]]:
                  flag = j
                  break
              
              if flag == -1:
                prev_status[s] = None
              
              else:
                se[0] = qu[flag]
                if flag == len(qu) - 1:
                  qu = []
                else:
                  qu = qu[flag+1:]
                prev_status[s] = [se, qu]

        all_queues_q_and_s.append(prev_status)
        
        min_so_far = 10000
        flag = -2
        min_idx = -1
        for s in range(len(prev_status)):
          if prev_status[s] == None:
            se = []
            qu = []
            se.append(i)
            all_queues_q_and_s[i][s] = [se, qu]
            flag = -3
            start_time_server[s] = arrival_times[i]
            end_time_server[s] = arrival_times[i] + service_times[i]
            start_time_customer[i] = arrival_times[i]
            end_time_customer[i] = arrival_times[i] + service_times[i]
            break
          else:
            se, qu = prev_status[s]
            if min_so_far < (len(se) + len(qu)):
              min_so_far = len(se) + len(qu)
              min_idx = s
              flag = -2
        
        if flag != -3:
          se, qu =all_queues_q_and_s[i][min_idx]
          qu.append(i)
          all_queues_q_and_s[i][min_idx] = [se, qu]
          start_time_customer[i] = end_time_customer[qu[len(qu)-2]]
          end_time_customer[i] = start_time_customer[i] + service_times[i]
        
      
    req_customer = len(all_queues_q_and_s)
    wait_time_customer = [0 for i in range(req_customer)]
    for i in range(req_customer):
      wait_time_customer[i] = abs(end_time_customer[i] - arrival_times[i])

    for idx in range(req_customer):
      for_each_lambda_departure[idx] = for_each_lambda_departure[idx] + wait_time_customer[idx]

  for q in range(N+20):
    for_each_lambda_departure[q] = for_each_lambda_departure[q] / total_iter
  
  super_average_2.append(for_each_lambda_departure)


W_l1_1 = super_average_1[0]
W_l2_1 = super_average_1[1]
W_l3_1 = super_average_1[2]
W_l4_1 = super_average_1[3]

W_l1_2 = super_average_2[0]
W_l2_2 = super_average_2[1]
W_l3_2 = super_average_2[2]
W_l4_2 = super_average_2[3]


fig = plt.figure()

sort_data_1_1 = np.sort(W_l1_1)
yvals_1_1 = np.arange(len(sort_data_1_1)) / float(len(sort_data_1_1)-1)
sort_data_2_1 = np.sort(W_l2_1)
yvals_2_1 = np.arange(len(sort_data_2_1)) / float(len(sort_data_2_1)-1)
sort_data_3_1 = np.sort(W_l3_1)
yvals_3_1 = np.arange(len(sort_data_3_1)) / float(len(sort_data_3_1)-1)
sort_data_4_1 = np.sort(W_l4_1)
yvals_4_1 = np.arange(len(sort_data_4_1)) / float(len(sort_data_4_1)-1)

sort_data_1_2 = np.sort(W_l1_2)
yvals_1_2 = np.arange(len(sort_data_1_2)) / float(len(sort_data_1_2)-1)
sort_data_2_2 = np.sort(W_l2_2)
yvals_2_2 = np.arange(len(sort_data_2_2)) / float(len(sort_data_2_2)-1)
sort_data_3_2 = np.sort(W_l3_2)
yvals_3_2 = np.arange(len(sort_data_3_2)) / float(len(sort_data_3_2)-1)
sort_data_4_2 = np.sort(W_l4_2)
yvals_4_2 = np.arange(len(sort_data_4_2)) / float(len(sort_data_4_2)-1)

x1 = np.arange(30)
y1 = np.exp(-(1-.7)*x1)
x2 = np.arange(40)
y2 = np.exp(-(1-.8)*x2)
x3 = np.arange(70)
y3 = np.exp(-(1-0.9)*x3)
x4 = np.arange(70)
y4 = np.exp(-(1-.95)*x4)

plt.plot(sort_data_1_1, 1-yvals_1_1, 'b', sort_data_2_1, 1-yvals_2_1, 'g', sort_data_3_1, 1-yvals_3_1, 'r', sort_data_4_1, 1-yvals_4_1, 'c',)
plt.plot(sort_data_1_2, 1-yvals_1_2, 'b', sort_data_2_2, 1-yvals_2_2, 'g', sort_data_3_2, 1-yvals_3_2, 'r', sort_data_4_2, 1-yvals_4_2, 'c',)
plt.plot(x1, y1, 'b', x2, y2, 'g', x3, y3, 'r', x4, y4, 'c',)

plt.yscale('log')
plt.xlabel('waiting time of customers - both policies')
plt.ylabel('Probability on a log(base-10) scale')
fig.savefig("Plot1.png")
plt.show()

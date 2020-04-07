import numpy as np
from numpy import array
import matplotlib.pyplot as plt

lam = [0.7, 0.8, 0.9, 0.95]
mu = 1
N = 10090
total_iter = 10
super_average = []
super_status = []

for la in range(len(lam)):
    
    super_average_lambda_arrival = []
    super_avergae_lambda_departure = []
    for_each_lambda_status = []

    for_each_lambda_arrival = [0 for i in range(N+20)]
    for_each_lambda_departure = [0 for i in range(N+20)]
    for_each_lambda_status = [0 for i in range(N+20)]
    
    for k in range(total_iter):
        
        service_times = []                                        #service_time[i]
        arrival_times = []                                        #arrival_time[i]  
        start_times_customers = []                                #start_time[i]
        end_time_customers = []                                   #end_time[i]
        queue_server_customer_number = []                         #[[S], [Q]]
        
        wait_time_customers = []                                  #W()
        number_of_customer_on_arrival = []                        #X(t)
        number_of_customer_on_departure = []                      #D(t)
        server_state = []                                         #Ns(t)
        
        
        l = lam[la]
        
        service_times.append(-1*np.log(1 - np.random.uniform())/mu)
        arrival_times.append(-1*np.log(1 - np.random.uniform())/l)

        for i in range(1, N):
            service_times.append(-1*np.log(1 - np.random.uniform())/mu)
            arrival_times.append(arrival_times[i-1] + (-1*np.log(1 - np.random.uniform())/l))

        for i in range(N):
            if i == 0:
                start_times_customers.append(arrival_times[0])
                end_time_customers.append(arrival_times[0] + service_times[0])
            else:
                start_times_customers.append(max(arrival_times[i], end_time_customers[i-1]))
                end_time_customers.append(start_times_customers[i] + service_times[i])    

        for i in range(N):
            server = []
            queue = []
            if i == 0:
                server.append(i)
                queue = []
                queue_server_customer_number.append([server, queue])
            else:
                flag = -1
                for j in range(i):
                    if arrival_times[i] >= start_times_customers[j] and arrival_times[i] <= end_time_customers[j]:
                        flag=j
                        break
                if flag==-1:
                    server.append(i)
                    queue = []
                else:
                    server.append(flag)
                    for t in range(flag+1, i+1):
                        queue.append(t)

                if arrival_times[i] >= end_time_customers[10000]:
                    queue_server_customer_number.append([server, queue])
                    break

                queue_server_customer_number.append([server, queue])    

        req_customer = len(queue_server_customer_number)

        for i in range(req_customer):
            wait_time_customers.append(end_time_customers[i] - arrival_times[i])

        for i in range(req_customer):
            s, q = queue_server_customer_number[i]
            if i in s:
                number_of_customer_on_arrival.append(0)
            elif i in q:
                number_of_customer_on_arrival.append(len(q))

        for i in range(req_customer):
            a = arrival_times[i]
            e = end_time_customers[i]

            ctr = 0

            for j in range(req_customer):
                if arrival_times[j] >= a and arrival_times[j] < e:
                    ctr = ctr + 1
                if arrival_times[i] >= e:
                    break

            number_of_customer_on_departure.append(ctr)

        for i in range(req_customer):
            if end_time_customers[i] < arrival_times[i+1]:
                server_state.append(0)
            else:
                server_state.append(1)
        
        for x in range(req_customer):
            for_each_lambda_arrival[x] = for_each_lambda_arrival[x] + number_of_customer_on_arrival[x]
            for_each_lambda_departure[x] = for_each_lambda_departure[x] + number_of_customer_on_departure[x]
            for_each_lambda_status[x] = for_each_lambda_status[x] + server_state[x]

    for q in range(N+20):
        for_each_lambda_arrival[q] = for_each_lambda_arrival[q] / total_iter
        for_each_lambda_departure[q] = for_each_lambda_departure[q] / total_iter
        for_each_lambda_status[q] = for_each_lambda_status[q] / total_iter

    super_average.append([for_each_lambda_arrival, for_each_lambda_departure])
    super_status.append(for_each_lambda_status)
        
        
X_l1 = super_average[0][0]  
X_l2 = super_average[1][0]  
X_l3 = super_average[2][0]  
X_l4 = super_average[3][0]

s_l1 = super_status[0]
s_l2 = super_status[1]
s_l3 = super_status[2]
s_l4 = super_status[3]

D_l1 = super_average[0][1]
D_l2 = super_average[1][1]
D_l3 = super_average[2][1]
D_l4 = super_average[3][1]

fig = plt.figure()

x1 = np.arange(20)
x2= np.arange(20)
x3 = np.arange(20)
x4 = np.arange(20)
lam_1 = [(lam[0])**i for i in x1]
lam_2 = [(lam[1])**i for i in x2]
lam_3 = [(lam[2])**i for i in x3]
lam_4 = [(lam[3])**i for i in x4]

sort_X_1 = np.sort(X_l1)
yvals_X_1 = np.arange(len(sort_X_1)) / float(len(sort_X_1)-1)
sort_X_2 = np.sort(X_l2)
yvals_X_2 = np.arange(len(sort_X_2)) / float(len(sort_X_2)-1)
sort_X_3 = np.sort(X_l3)
yvals_X_3 = np.arange(len(sort_X_3)) / float(len(sort_X_3)-1)
sort_X_4 = np.sort(X_l4)
yvals_X_4 = np.arange(len(sort_X_4)) / float(len(sort_X_4)-1)

sort_D_1 = np.sort(D_l1)
yvals_D_1 = np.arange(len(sort_D_1)) / float(len(sort_D_1)-1)
sort_D_2 = np.sort(D_l2)
yvals_D_2 = np.arange(len(sort_D_2)) / float(len(sort_D_2)-1)
sort_D_3 = np.sort(D_l3)
yvals_D_3 = np.arange(len(sort_D_3)) / float(len(sort_D_3)-1)
sort_D_4 = np.sort(D_l4)
yvals_D_4 = np.arange(len(sort_D_4)) / float(len(sort_D_4)-1)

sort_s_1 = np.sort(s_l1)
yvals_s_1 = np.arange(len(sort_s_1)) / float(len(sort_s_1)-1)
sort_s_2 = np.sort(s_l2)
yvals_s_2 = np.arange(len(sort_s_2)) / float(len(sort_s_2)-1)
sort_s_3 = np.sort(s_l3)
yvals_s_3 = np.arange(len(sort_s_3)) / float(len(sort_s_3)-1)
sort_s_4 = np.sort(s_l4)
yvals_s_4 = np.arange(len(sort_s_4)) / float(len(sort_s_4)-1)

plt.plot(x1, lam_1, 'b', x1, lam_2, 'g', x1, lam_3, 'r', x1, lam_4, 'c')
plt.plot(sort_X_1, 1-yvals_X_1, 'b', sort_X_2, 1-yvals_X_2, 'g', sort_X_3, 1-yvals_X_3, 'r', sort_X_4, 1-yvals_X_4, 'c',)
plt.plot(sort_D_1, 1-yvals_D_1, 'b', sort_D_2, 1-yvals_D_2, 'g', sort_D_3, 1-yvals_D_3, 'r', sort_D_4, 1-yvals_D_4, 'c')

plt.yscale('log')
plt.xlabel('number of customers as seen on arrivals and departures')
plt.ylabel('Probability on a log(base-10) scale')
fig.savefig("Plot1.png")

fig2 = plt.figure()
plt.plot(sort_s_1, 1-yvals_s_1, 'b', sort_s_2, 1-yvals_s_2, 'g',sort_s_3, 1-yvals_s_3, 'r', sort_s_4, 1-yvals_s_4, 'c')
plt.yscale('log')
plt.xlabel('Events - busy/idle periods')
plt.ylabel('CCDF - emperical')
fig2.savefig("Plot2.png")

plt.show()

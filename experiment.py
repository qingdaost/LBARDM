import numpy as np
import helper
import statistics
import copy
from numpy import random
import time

random.seed(0)

latency = helper.latencymap
cloud = helper.datacenter
request = helper.request
Tset = helper.workflowset
datalocation = 26  #East, USA

POP_SIZE = 100  # population size
CROSS_RATE = 0.9  # mating probability
MUTATION_RATE = 0.1  # mutation probability
N_GENERATIONS = 50
capacity = [1, 2, 4, 8, 16, 48, 64, 96]
budget_rate = [0.1, 0.2, 0.3, 0.4, 0.7]
run = 30

# For each workflow
for i in range(len(Tset)):
    print("*" * 20)
    T = Tset[i]
    print("Workflow", i + 1)
    services_SIZE = len(T)            # DNA length
    print("*" * 20)

    # For each budget factor
    for rate in budget_rate:
        # calculate budget
        high = len(T) * cloud[12][2][-1]   #Sao Paulo, Brazil
        baseline = [[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
        vmtype = []
        low = 0
        for s in T:
            for i in range(len(capacity)):
                if (1000 / s[2]) * capacity[i] > sum(request):
                    break;
            vmtype.append(i)
            low += cloud[0][2][i]   #East, USA
        print(low, high)
        baseline.append(type)
        budget = low + rate * (high - low)
        print("budget:", rate, budget)
        print("*" * 20)

        # Distribute requests among replicas
        def get_distribution(ind):
            distribution = []
            k = 0
            for i in range(82):
                upperbound = 1000
                for j in range(15):
                    if ind[j] == 1 and latency[i][cloud[j][0]] < upperbound:
                        upperbound = latency[i][cloud[j][0]]
                        k = j
                distribution.append(k)
            #print(distribution)
            return distribution

        # Calculate workload
        def get_load(ind, distribution):
            load = []
            for i in range(15):
                temp = 0
                if ind[i] == 1:
                    for j in range(82):
                        if distribution[j] == i:
                            temp += request[j]
                load.append(temp)
            #print(load)
            return load

        # Calculate network latency
        def get_nl(ind, distribution, load):
            anl = []
            for i in range(15):
                temp = 0
                if ind[0][i] == 1 and load[i] != 0:
                    for j in range(82):
                         if distribution[j] == i:
                            temp += request[j] * latency[j][cloud[i][0]]
                    anl.append(temp / load[i])
                else:
                    anl.append(temp)
            #print(anl)
            return anl

        # Calculate makespan
        def get_wrt(ind, load):
            wrt = []
            flag = 0
            for d in range(15):
                rt = 0
                if ind[0][d] == 1:
                    estlist = []
                    for i in range(len(T)):
                        est = 0
                        if T[i][0] != []:
                            for j in T[i][0]:
                                s = T[j - 1][2]
                                #print(services_SIZE * flag + j - 1)
                                type = ind[1][services_SIZE * flag + j - 1]
                                # data access service 22 and 24
                                if s == 22 or s == 24:
                                    s += latency[cloud[d][0]][datalocation]
                                if 1000 / s * capacity[type] > load[d]:
                                    temp = estlist[j - 1] + 1 / (1000 / s * capacity[type] - load[d]) * 1000
                                else:
                                    temp = float("inf")
                                    #print("Overload: ", T[j - 1])
                                if temp > est:
                                    est = temp
                        estlist.append(est)
                        #print(estlist)
                    s = T[-1][2]
                    #print(services_SIZE * flag + services_SIZE - 1)
                    type = ind[1][services_SIZE * flag + services_SIZE - 1]
                    # data access service 22 and 24
                    if s == 22 or s == 24:
                        s += latency[cloud[d][0]][datalocation]
                    if 1000 / s * capacity[type] > load[d]:
                        rt = estlist[-1] + 1 / (1000 / s * capacity[type] - load[d]) * 1000
                    else:
                        rt = float("inf")
                        #print("Overload")
                    flag += 1
                wrt.append(rt)
            #print("Response time: ", wrt)
            return wrt
        #get_wrt(test, T, load)

        # Calculate response time
        def get_rt(ind):
            distribution = get_distribution(ind[0])
            load = get_load(ind[0], distribution)
            wrt = get_wrt(ind, load)
            nl = get_nl(ind, distribution, load)
            #print(wrt, nl)
            count = 0
            for i in range(15):
                count += (wrt[i] + 2 * nl[i]) * load[i] #* rateset[i]
            art = count / sum(request)
            #print("Response time: ", art)
            return art


        # Calculate deployment cost
        def get_cost(ind):
            cost = 0
            for i in range(len(ind[1])):
                index = get_copyindex(ind, i)
                #print(cloud[location][2][ind[1][i]])
                cost += cloud[index][2][ind[1][i]]
            #print("Cost: ", cost)
            return cost

        # mapping datacenter
        def get_copyindex(ind, typeindex):
            #print(ind)
            copy = int(typeindex / services_SIZE)
            index = [i for i, n in enumerate(ind[0]) if n == 1][copy]
            return index


        # Generate a random replication plan
        def random_replicaplan():
            replicaplan = []
            for i in range(15):
                replicaplan.append(random.randint(0, 2))
            return replicaplan

        # generate cheapest feasible solution
        def generate_cheapestplan(replicaplan, load):
            type = []
            for i in range(15):
                if replicaplan[i] == 1:
                    for j in range(services_SIZE):
                        temp = 0
                        s = T[j][2]
                        if s == 22 or s == 24:
                            s += latency[cloud[i][0]][datalocation]
                        while temp < 8 and 1000 / s * capacity[temp] <= load[i]:
                            temp += 1
                        if temp > 7:
                            type.append(7)
                        else:
                            type.append(temp)
            cheapestplan =[replicaplan]
            cheapestplan.append(type)
            return cheapestplan


        # roulette-wheel selection
        def select(pop, fitness):  # nature selection wrt pop's fitness
            idx = list(np.random.choice(np.arange(POP_SIZE), size=POP_SIZE, replace=True,
                                   p=fitness / fitness.sum()))
            newpop = []
            for i in idx:
                newpop.append(pop[i])
            return newpop


        def crossover(parent, pop):  # mating process (genes crossover)
            if np.random.rand() < CROSS_RATE:
                i_ = random.randint(0, POP_SIZE)  # select another individual from pop
                cross_point = np.random.randint(0, 15)  # choose one crossover point
                replicaplan = parent[0][:cross_point]
                replicaplan += pop[i_][0][cross_point:]
            else:
                replicaplan = parent[0]
            return replicaplan


        def mutate(replicaplan):
            oldplan = replicaplan[:]
            for point in range(15):
                if np.random.rand() < MUTATION_RATE:
                    oldplan[point] = 1 if oldplan[point] == 0 else 0
            replicaplan = oldplan
            child = gain_solution(replicaplan)
            if child[0].count(1) * services_SIZE != len(child[1]):
                print("Error", child)
            return child

        # Add a replica in a specific datacenter
        def addcopy_specify(solution, index):
            temp = copy.deepcopy(solution)
            temp[0][index] = 1
            replicaplan = temp[0]
            solution = gain_solution(replicaplan)
            return solution

        # Gain-based VM deployment
        def gain_solution(replicaplan):
            distribution = get_distribution(replicaplan)
            load = get_load(replicaplan, distribution)
            greedy = generate_cheapestplan(replicaplan, load)
            cost = get_cost(greedy)
            rt = get_rt(greedy)
            while cost <= budget:
                # print("next")
                solutionset = []
                gainset = []
                for i in range(len(greedy[1])):
                    if greedy[1][i] < 7:
                        temp = copy.deepcopy(greedy)
                        temp[1][i] = greedy[1][i] + 1

                        newcost = get_cost(temp)
                        if newcost <= budget:
                            # print(temp)
                            newrt = get_rt(temp)
                            gain = (rt - newrt) / (newcost - cost)
                            # print(gain)
                            if gain > 0:
                                solutionset.append(temp)
                                gainset.append(gain)
                if len(gainset) == 0:
                    break
                else:
                    greedy = solutionset[np.argmax(gainset)]
                    # print("473", solution)
                    cost = get_cost(greedy)
                    rt = get_rt(greedy)
                    # print(rt)
            #print(seed, get_cost(seed), get_rt(seed))
            return greedy

        # Experiments

        # NearData
        # Dataset is in Virginia (US East)
        replicaplan = [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        datasolution = gain_solution(replicaplan)
        cost = get_cost(datasolution)
        rt = get_rt(datasolution)
        print("NearData:", datasolution, cost, rt)

        # NearUsers
        replicaplan = [1] * 15
        usersolution = gain_solution(replicaplan)
        cost = get_cost(usersolution)
        rt = get_rt(usersolution)
        if cost <= budget:
            print("NearUsers:", usersolution, cost, rt)
        else:
            print("NearUsers: N.A.")

        # Greedy-Gain as seed
        seed = gain_solution([1,0,0,0,0,0,0,0,0,0,0,0,0,0,0]) # one cheapest replica
        while seed[0].count(1) < 15:
            candidatesolution = []
            improveset = []
            temp = copy.deepcopy(seed)
            for i in range(15):
                if temp[0][i] == 0:
                    newsolution = addcopy_specify(temp, i)
                    if get_cost(newsolution) <= budget:
                        improve = rt - get_rt(newsolution)
                        if improve > 0:
                            candidatesolution.append(newsolution)
                            improveset.append(improve)
            if len(improveset) == 0:
                break
            else:
                seed = candidatesolution[np.argmax(improveset)]
                cost = get_cost(seed)
                rt = get_rt(seed)
                print("Better solution: ", seed[0].count(1), seed, cost, rt)

        print("Greedy-Gain: ", seed[0].count(1), seed, get_cost(seed), get_rt(seed))


        #GA
        start = time.time()
        costresult = []
        timeresult = []

        for t in range(run):
            #print(baseline, get_rt(baseline), get_cost(baseline))

            print("=" * 20)
            print("GA run: ", t)

            pop = []
            rtlist = []

            # seed
            pop.append(seed)
            rtlist.append(get_rt(seed))

            # randomness
            count = 0
            while count < POP_SIZE - 1:
                replicaplan = random_replicaplan()
                if replicaplan.count(1) > 0:
                    ind = gain_solution(replicaplan)
                    seedcost = get_cost(ind)
                    if seedcost <= budget:
                        pop.append(ind)
                        rt = get_rt(ind)
                        rtlist.append(rt)
                        count += 1

            # Loop
            best = min(rtlist)
            elite = copy.deepcopy(pop[np.argmin(rtlist)])
            print("Elite:", elite, best, get_cost(elite))
            log = [best]

            for n in range(N_GENERATIONS):

                count = 0
                for i in range(len(pop)):
                    ind = pop[i]
                    # print(ind)
                    cost = get_cost(ind)


                    for i in range(len(pop)):
                        ind = pop[i]
                        # print(ind)
                        cost = get_cost(ind)
                        # print(cost)
                        # death penalty
                        if cost > budget or ind[0].count(1) == 0:
                            # print("change")
                            pop[i] = elite
                            count += 1
                            rtlist[i] = best
                        else:
                            rtlist[i] = get_rt(ind)

                # print(count/POP_SIZE)
                # check performance
                newbest = pop[np.argmin(rtlist)]
                newrt = min(rtlist)
                log.append(newrt)
                # print("New best:", newbest, min(rtlist), get_cost(newbest))

                # Elite
                elite = copy.deepcopy(newbest)
                if newrt < best:
                    print("GENERATION:", n)
                    best = newrt
                    print("New elite:", elite, get_rt(elite), get_cost(elite))

                fitness = np.divide(1, np.array(rtlist))
                # print(fitness)

                pop = select(pop, fitness)
                # print(pop)
                pop_copy = pop.copy()

                pop[0] = elite
                for i in range(1, POP_SIZE):
                    temp = crossover(pop[i], pop_copy)
                    # child = mutate(pop[i])
                    child = mutate(temp)
                    pop[i] = child  # parent is replaced by its child

                #print(len(memory))

            print("Final solution:", elite, best, get_cost(elite))
            print(log)
            costresult.append(get_cost(elite))
            timeresult.append(best)

        print(costresult)
        print(timeresult)
        
        print("*" * 20)
        print('cost mean:', sum(costresult) / run)
        print('cost sd:', statistics.stdev(costresult))
        print('time mean:', sum(timeresult) / run)
        print('time sd:', statistics.stdev(timeresult))

        end = time.time()
        print((end - start) / run)

import numpy as np

def random_learning(net,loss_fun,max_its,max_mutations,step,threshold):
    '''
    returns loss over iterations,
    (iterations reached,total mutation attempts)
    '''
    losses = [loss_fun(net)]
    total_k = 0

    for i in range(max_its):
        if losses[-1] < threshold:
            break
        
        for k in range(max_mutations): #mutate so many times before giving up
            pertubations = random_mutate(net,step)
            loss = loss_fun(net)

            if loss < losses[-1]:
                losses.append(loss)
                break
            else:
                #pertubate in the opposite direction to save generating another pertubation
                undo_mutate(net,pertubations)
                undo_mutate(net,pertubations)

            loss = loss_fun(net)
            if loss < losses[-1]:
                losses.append(loss)
                break
            else:
                make_mutate(net,pertubations)
            
        total_k += k
        if k == max_mutations-1:
            break

    return losses,(i,total_k)

def random_mutate(net,step): #randomly mutate the network
    pertubations = []
    for index in net.mutateable_layers:
        w_shape = np.shape(net.layers[index].weights)
        b_shape = np.shape(net.layers[index].biases)

        w_pertubation = step * np.random.normal(size = w_shape)
        b_pertubation = step * np.random.normal(size = b_shape)

        pertubations.append([w_pertubation,b_pertubation])

        net.layers[index].weights += w_pertubation
        net.layers[index].biases += b_pertubation

    return pertubations

def undo_mutate(net,pertubations):
    i = 0
    for index in net.mutateable_layers:
        w_pertubation = pertubations[i][0]
        b_pertubation = pertubations[i][1]

        net.layers[index].weights -= w_pertubation
        net.layers[index].biases -= b_pertubation

        i += 1

def make_mutate(net,pertubations):
    i = 0
    for index in net.mutateable_layers:
        w_pertubation = pertubations[i][0]
        b_pertubation = pertubations[i][1]

        net.layers[index].weights += w_pertubation
        net.layers[index].biases += b_pertubation

        i += 1
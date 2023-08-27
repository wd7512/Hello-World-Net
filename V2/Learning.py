import numpy as np

def random_learning(net,loss_fun,max_its,max_mutations,step,threshold):
    losses = [loss_fun(net)]

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
                undo_mutate(net,pertubations)

        if k == max_mutations:
            break

    return losses

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

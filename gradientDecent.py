
# batch gradient descent func
def batchGD(set, w0, w1, lr):
    # preparing variables
    loops = 0
    maxLoops = 10 ** 6
    diff = 1
    convg = 10 ** -10
    totalHw0 = 0
    totalHw1 = 0
    # looping through until convergence or maxloop count is hit
    while loops < maxLoops and diff > convg:
        for i in range(len(set)):   # for each pair of coords in the set
            coord = set[i]          # get coord pair
            Y = coord[1]            # get Y
            X = coord[0]            # get X
            Hwx = w0 + (w1 * X)     # calc update values
            totalHw0 += Y - Hwx     # and sum them separately
            totalHw1 += (Y - Hwx) * X
        # save new update values as another variable to get difference from old ones
        nw0 = w0 + (lr * totalHw0)
        nw1 = w1 + (lr * totalHw1)
        # checking convergence
        diff = min(abs((nw0 - w0)), abs((nw1 - w1)))
        # then overwrite old values, and re-loop
        w0 = nw0
        w1 = nw1
        loops += 1
    # final weight values from the gradient decent
    result = f"Y = {w0} + {w1}*X"
    return f'batchGDs result: {result}, found in {loops} loops'

# stochastic gradient descent func
def stochGD(set, w0, w1, lr):
    # preparing variables
    loops = 0
    maxLoops = 10 ** 6
    diff = 1
    i = 0
    convg = 10 ** -10
    # looping through until convergence or maxloop count is hit
    while loops < maxLoops and diff > convg:
        if i == len(set):    # loop index while looping, prevent index overflow error
            i = 0
        coord = set[i]       # get coord pair
        Y = coord[1]         # get Y
        X = coord[0]         # get X
        Hwx = w0 + (w1 * X)  # calc update value
        # save new update values as another variable to get difference from old ones
        nw0 = w0 + (lr * (Y - Hwx))
        nw1 = w1 + (lr * ((Y - Hwx) * X))
        # checking convergence
        diff = min(abs((nw0 - w0)), abs((nw1 - w1)))
        # then overwrite old values, and re-loop
        w0 = nw0
        w1 = nw1
        i += 1
        loops += 1
    # final weight values from the gradient decent
    result = f"Y = {w0} + {w1}*X"
    return f'stochGDs result: {result}, found in {loops} loops'


dataset = [(2, 5), (4, 7), (6, 14), (7, 14), (8, 17), (10, 19)]
print(batchGD(dataset, 0.25, 0.25, 0.0000001))
print(stochGD(dataset, 0.25, 0.25, 0.0001))

def interpolate(arr, x, y):
    H = arr.shape[0]
    W = arr.shape[1]
    
    weighted = arr[int(y), int(x)] * (1 - (x - int(x))) * (1 - (y - int(y))) + \
               arr[int(y), int(x) + 1] * (x - int(x)) * (1 - (y - int(y))) + \
               arr[int(y) + 1, int(x)] * (1 - (x - int(x))) * (y - int(y)) + \
               arr[int(y) + 1, int(x) + 1] * (x - int(x)) * (y - int(y))
    
    return weighted
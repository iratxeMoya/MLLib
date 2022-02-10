
from numpy import True_


def filterStrs(elem, data):
    try:
        elem = float(data[elem][0])
        return True

    except:
        return False

def filterByName(elem, notFeatures):
    
    if elem not in notFeatures:
        return True
    else:
        return False
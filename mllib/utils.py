
def filterStrs(elem, data):
    try:
        elem = float(data[elem][0])
        return True

    except:
        return False
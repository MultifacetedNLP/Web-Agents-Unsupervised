import collections, numpy

def synthesize(array):
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    if len(array) > 0:
        d["min"] = numpy.amin(array)
        d["max"] = numpy.amax(array)
    else:
        d["min"] = numpy.nan
        d["max"] = numpy.nan
    return d
import numpy as np
from sklearn.metrics import r2_score

quarter = np.array([1, 2, 3, 4, 5, 6, 7, 8])

# revenue
ibm_rev = np.array([17560, 1925, 13187, 18745, 17618, 16694, 14197, 15535])
ibm_last = 14107
msft_rev = np.array([35021, 38033, 37154, 43076, 41706, 46152, 45317, 51728])
msft_last = 49360
aapl_rev = np.array([59685, 64698, 111439, 89584, 81494, 83360, 123945, 97278])
aapl_last = 82959
goog_rev = np.array([38297, 46173, 56898, 55314, 61887, 65056, 75122, 67733])
fb_reb = np.array([18687, 21470, 28071, 26171, 29077, 29010, 33671, 27908])
pg_rev = np.array([19318, 19745, 18109, 18946, 20338, 20953, 19381, 19515])
ge_rev = np.array([16805, 18529, 21009, 17071, 18157, 18592, 20495, 17036])


# build linear model and get r2
def getLinear(data):
    fit = np.polyfit(quarter, data, 1)
    a = fit[0]
    b = fit[1]
    fitted = (quarter * a) + b
    prediction = (9 * a) + b
    return r2_score(data, fitted)

# build log model and get r2
def getLog(data):
    fit = np.polyfit(np.log(quarter), data, 1)
    a = fit[0]
    b = fit[1]
    fitted = (a*np.log(quarter)) + b
    prediction = (a*np.log(9)) + b
    return r2_score(data, fitted)

# build exponential model and get r2
def getExp(data):
    fit = np.polyfit(quarter, np.log(data), 1)
    a = np.exp(fit[1])
    b = fit[0]
    fitted = a * np.exp(b * quarter)
    prediction = a * np.exp(b * 9)
    return r2_score(data, fitted)

# build power curve model and get r2
def getPower(data):
    fit = np.polyfit(np.log(quarter), np.log(data), 1)
    a = np.exp(fit[1])
    b = fit[0]
    fitted = a * np.power(quarter, b)
    prediction = a * np.power(9, b)
    return r2_score(data, fitted)

def bestModel(data):
    curr = 0
    model = ""

    r2 = getLinear(data)
    if(r2 > curr):
        curr = getLinear(data)
        model = "linear"

    r2 = getLog(data)
    if(r2 > curr):
        curr = getLog(data)
        model = "log"

    r2 = getExp(data)
    if(r2 > curr):
        curr = getExp(data)
        model = "exponential"

    r2 = getPower(data)
    if(r2 > curr):
        curr = getPower(data)
        model = "power"

    return model, curr

def main():
    model, curr = bestModel(ibm_rev)
    print("IBM Revenue: " + model + " " + str(curr))

    model, curr = bestModel(msft_rev)
    print("MSFT Revenue: " + model + " " + str(curr))

    model, curr = bestModel(aapl_rev)
    print("AAPL Revenue: " + model + " " + str(curr))

    model, curr = bestModel(goog_rev)
    print("GOOG Revenue: " + model + " " + str(curr))

    model, curr = bestModel(fb_reb)
    print("FB Revenue: " + model + " " + str(curr))

    model, curr = bestModel(pg_rev)
    print("PG Revenue: " + model + " " + str(curr))

    model, curr = bestModel(ge_rev)
    print("GE Revenue: " + model + " " + str(curr))


main()


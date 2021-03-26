import csv

def SaveModel(weights, filename):
    with open(filename + '.csv', mode='w') as infile:
        writer = csv.DictWriter(infile, ['stateHash', 'stateValue'])
        writer.writeheader()
        for weight in weights:
            writer.writerow({'weights': weight})

def LoadModel(fileName):
    weights = []
    try:
        with open(fileName + '.csv', mode='r') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                weights.append(float(row['weights']))
    except Exception as e:
        print(e)
    return weights
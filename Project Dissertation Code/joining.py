
with open('Reddit-critical-data.csv', 'r') as f1:
    original = f1.read()

with open('Reddit-noncritical-data.csv', 'a') as f2:
    f2.write('\n')
    f2.write(original)

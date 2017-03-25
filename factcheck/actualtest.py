import factcheck

while (True):
    inp = input('Enter a sentence: ')

    if inp == 'q':
        exit()

    print(factcheck.factAnalysis(inp))
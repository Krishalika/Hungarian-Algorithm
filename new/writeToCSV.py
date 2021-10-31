from __future__ import print_function
import csv  
import eel

eel.init('web', allowed_extensions=['.js', '.html'])

@eel.expose #expose this function to Javascript
def writeToCSV (data):
    csv_file = open ('./data1.csv', 'w')
    csv_file.write(data)
    csv_file.close()

eel.start('table.html',size=(600,600)) #start
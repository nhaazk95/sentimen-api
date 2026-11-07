import csv

with open('test.csv', 'r', encoding='utf-8') as f_in, \
     open('hasil_tanpa_label.csv', 'w', encoding='utf-8', newline='') as f_out:
    
    reader = csv.DictReader(f_in)
    fieldnames = [c for c in reader.fieldnames if c != 'label']
    
    writer = csv.DictWriter(f_out, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    
    for row in reader:
        row.pop('label', None)
        writer.writerow(row)

print("Selesai! Cek file: hasil_tanpa_label.csv")
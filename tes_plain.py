import main
s = ('Pembuka.\n\n1. RS Satu (Kec A)\n- Lokasi: x\n- Kata pasien: \u201cenak sekali\u201d\n\n'
     '2. RS Dua (Kec B)\n- Lokasi: y\n- Kata pasien: \u201cbagus\u201d\n\nKesimpulan: ok')
print(main.to_plain_text(s))
print('---')
print(main.to_plain_text(s, keep_tables=True))
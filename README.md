# wordle_stuff

## wordle_helper.py
`wordle_helper.py` helps you beat wordle via parameterized rules (see `wordle_helper.py` for full description of functionality).  
For example if the I initally guess "BEAST" and the target word is "ADORN":
"A" is in the wrong place as the third letter. "B," "E," "S," and "T" are not present. So we run:
```
python wordle_helper.py --wrong_place a3 --not_present best
```
This outputs:  
Guess: Score  
CAIRN: 5.102  
DINAR: 5.101  
NADIR: 5.101  
CAIRO: 5.1  
RADIO: 5.099  

If we guess "CAIRN," we have "A" in the wrong place again, add "C" and "I" to the list of letters not present, and "R" and "N" in the correct places. With short flags:
```
python wordle_helper.py -c r4 n5 -p a3 a2 -n bestci
```
Guess: Score  
ADORN: 5.096  

We get the answer with 3 attempts!

Credit for corncob_caps.txt: https://archive.ph/tDYtI  
This and other word lists are conveniently available here: https://github.com/yelsharawy/WordBlitzer/tree/main

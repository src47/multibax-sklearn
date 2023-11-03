#!/usr/bin/python3

import sys

def main():
    if len(sys.argv)<2:
        print("syntax: HelloWorld <your name>")
        return
    n = ' '.join(sys.argv[1:])
    print("Hey there %s.  What's shakin'?"%(n))

    return

if __name__=='__main__':
    main()

#!/Users/alexstein/anaconda3/bin/python

# revtr_preprocessing.py
# - parse a single VP CSV file
# - filter out paths > 8 hops,
# - map the destination IP to (asn,bgp-prefix) pair

import pyasn
import threading
import datetime
import time
import os
import sys
import csv

# the BGP-dump and destinfo dict must be global
# - these will be accessed concurrently by multiple threads
tag = datetime.datetime.now().strftime('%Y-%m-%d')
bgpdumpfile = 'bgpdumps/'+tag+'.dat'
asndb = pyasn.pyasn(bgpdumpfile)
destinfo = {}

# _numhops
# returns either the number of hops this ping takes to reach dst, or 10 if never
def _numhops(ping):
    dest = ping[0]
    dist = 0
    for hop in ping[1:]:
        dist = dist + 1
        if hop == dest:
            break
    return dist


# _lookuptask
# query the BGP-dump database with IP, and store value in destinfo[] dict
# no locking is needed! Python's core data structures are all thread-safe
# https://docs.python.org/3/glossary.html#term-global-interpreter-lock
def _lookuptask(ip):
    try:
        destinfo[ip] = asndb.lookup(ip)
    except:
        return

# _asninfo
# spin off a _lookuptask(ip) thread
# returns tuple(thread-handle, ip-address)
def _asninfo(ip):
    t = threading.Thread(target=_lookuptask(ip), name=ip)
    t.start()
    return (t, ip)

# FilterAndCount ...
# Remove pings from the input CSV file for which the destination is never reached
# Yields tuple(number-of-hops, ip-address)
# Adapted from StackOverflow post: https://stackoverflow.com/a/17444799/3341596
def FilterAndCount(filename):
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        # remove the header line
        next(datareader)
        # filter and count
        yield from map(lambda ping: (_numhops(ping), ping[0]),
            filter(lambda ping: _numhops(ping) < 9, datareader))
        return

# MapPrefixes ...
# Take filtered output (numhops, ipaddr) tuples and collect their asn info
def MapPrefixes(filename):
    cnts = FilterAndCount(filename)
    yield from map(lambda tup: (tup[0], _asninfo(tup[1])), cnts)
    return
    
min_dist_per_asn = {}
min_dist_per_prefix = {}
def FindMins(vp):
    for entry in MapPrefixes('vps/'+vp):
        # parse apart the returned entry
        hops = entry[0]         # number of hops to dest
        thread = entry[1][0]    # thread handle
        dest = entry[1][1]      # destination ip-address

        # wait for the generated thread to finish
        thread.join()

        # gather the output
        dinfo = destinfo[dest]
        asn = str(dinfo[0])     # ASN number
        prefix = str(dinfo[1])  # BGP prefix

        # collect the mins
        if asn == None or prefix == None:
            continue
        try:
            if hops < min_dist_per_asn[asn][0]:
                min_dist_per_asn[asn] = (hops,vp)
        except KeyError:
            min_dist_per_asn[asn] = (hops,vp)
        try:
            if hops < min_dist_per_prefix[prefix][0]:
                min_dist_per_prefix[prefix] = (hops,vp)
        except KeyError:
            min_dist_per_prefix[prefix] = (hops,vp)
    return

def main():
    start = time.time()
    for vp in os.listdir('vps/'):
        FindMins(vp)
    for asn, pair in sorted(min_dist_per_asn.items(), key=lambda k: k[1]):
        print("ASN: {: <6}, min_dist: {:d}, vp: {: <30}".format(asn, pair[0], pair[1]))
    end = time.time()
    print("time elapsed: {}".format(end-start)) 
    
if __name__ == "__main__":
    main()
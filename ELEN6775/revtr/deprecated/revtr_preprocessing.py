#!/Users/alexstein/anaconda3/bin/python

# revtr_preprocessing.py
# - parse all VP CSV files, one-by-one from the online source
# - filter out paths > 9 hops
# - map the destination IP to (asn,bgp-prefix) pair
# - find the minimum distance per-VP to each destination AS/prefix
# usage:
# - ./revtr_preprocessing.py <username:password>
# note:
# - be sure to update the BGP datadump before running this script
# - to do so, run ./scripts/refresh

# for data parsing
import pyasn
import threading
import datetime
import time
import json
import wget
import os
import sys
import csv

# for file download, etc.
import re, pycurl
from io import BytesIO
from bs4 import BeautifulSoup

#==========================================================
# BGP Processing Functions
#==========================================================

# username and password for cget is passed via command-line
try:
    userpwd = sys.argv[1]
except IndexError:
    raise ValueError('error: please enter username:password.')

# the BGP-dump and destinfo dict must be global
# - these will be accessed concurrently by multiple threads
tag = datetime.datetime.now().strftime('%Y-%m-%d')
bgpdumpfile = 'bgpdumps/'+tag+'.dat'
asndb = pyasn.pyasn(bgpdumpfile)

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
def _lookuptask(ip, destinfo):
    try:
        destinfo[ip] = asndb.lookup(ip)
    except:
        return

# _asninfo
# spin off a _lookuptask(ip) thread
# returns tuple(thread-handle, ip-address)
def _asninfo(ip, destinfo):
    t = threading.Thread(target=_lookuptask(ip,destinfo), name=ip)
    t.start()
    return (t, ip)

# FilterAndCount ...
# Remove pings from the input CSV file for which the destination is never reached
# Yields tuple(number-of-hops, ip-address, [ping])
# Adapted from StackOverflow post: https://stackoverflow.com/a/17444799/3341596
def FilterAndCount(vp):
    filename = cget(vp+'.csv')
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        # remove the header line
        next(datareader)
        # filter and count
        yield from map(lambda ping: (_numhops(ping), ping[0], ping),
            filter(lambda ping: _numhops(ping) <= 9, datareader))

    # these files are large, don't want to keep them around!
    os.remove(filename)
    return

# MapPrefixes ...
# Take filtered output (numhops, ipaddr, [ping]) tuples and collect their asn info
def MapPrefixes(vp,destinfo):
    cnts = FilterAndCount(vp)
    # tup[0] = numhops, tup[1] = destination IP address, tup[2] = the full measurement
    # the below line will yeild a 3-tuple (hops, IP, [ping])
    yield from map(lambda tup: (tup[0], _asninfo(tup[1],destinfo), tup[2]), cnts)
    return

# FindMins ...
# Record the minimum distance to asn/prefix per VP; keep the [ping] measurement around
min_dist_per_asn = {}
min_dist_per_prefix = {}
def FindMins(vp):
    destinfo = {}
    for entry in MapPrefixes(vp, destinfo):
        # parse apart the returned entry
        hops = entry[0]         # number of hops to dest
        thread = entry[1][0]    # thread handle
        dest = entry[1][1]      # destination ip-address
        measurement = entry[2]  # ping measurement

        # wait for the generated (updating destinfo) thread to finish
        thread.join()

        # gather the output
        if dest in destinfo:
            dinfo = destinfo[dest]
            asn = str(dinfo[0])     # ASN number
            prefix = str(dinfo[1])  # BGP prefix
        else:
            continue

        # collect the mins by ASN and Prefix
        if asn == None or prefix == None:
            continue

        if asn in min_dist_per_asn:
            if vp in min_dist_per_asn[asn]:
                if hops < int(min_dist_per_asn[asn][vp][0]):
                    min_dist_per_asn[asn][vp] = (str(hops), measurement)
            else:
                min_dist_per_asn[asn][vp] = (str(hops), measurement)
        else:
            min_dist_per_asn[asn] = {vp:(str(hops), measurement)}

        if prefix in min_dist_per_prefix:
            if vp in min_dist_per_prefix[prefix]:
                if hops < int(min_dist_per_prefix[prefix][vp][0]):
                    min_dist_per_prefix[prefix][vp] = (str(hops), measurement)
            else:
                min_dist_per_prefix[prefix][vp] = (str(hops), measurement)
        else:
            min_dist_per_prefix[prefix] = {vp:(str(hops), measurement)}

    return

#==========================================================
# File Processing Functions
#==========================================================

rooturl = "http://bgoodc.cs.columbia.edu/old_rr_data/"

# cget
# the python wget package has no options, so using curl instead
def cget(extension):
    filename = 'vps/data_'+extension
    with open(filename, 'wb') as f:
        url = rooturl + extension
        c = pycurl.Curl()
        c.setopt(c.URL, url)
        c.setopt(c.USERPWD, userpwd)
        c.setopt(c.WRITEDATA, f)
        c.perform()
        c.close()
    return filename

#==========================================================
# Main
#==========================================================

def main():
    start = time.time()

    # get and process the list of vp-csv files
    vplist = set()
    vplisthtml = cget("")
    soup = BeautifulSoup(open(vplisthtml,'r'),'html.parser')
    # num_to_read = 139 # only here while I write the scripts
    num_to_read = 5
    for hit in soup.find_all('a'):
        match = re.match(r'^.*csv', hit['href'])
        if match and num_to_read > 0:
            vplist.add(match[0])
            num_to_read = num_to_read - 1

    # done extracting, remove the vp-html-file
    os.remove(vplisthtml)

    for vpcsv in vplist:
        FindMins(os.path.splitext(vpcsv)[0]) # removes the ".csv"
        print("Processed " + vpcsv + "...")

    # sort and store minimums by ASN
    with open('mappings/min_by_asn.json', 'w') as asnfile:
        dump = {}
        for asn, rankings in min_dist_per_asn.items():
            dump[asn] = []
            dump[asn].append({
                'rankings': rankings
            })
        # dump json mappings out to asnfile     
        json.dump(dump, asnfile)

    # sort and store minimums by Prefix
    with open('mappings/min_by_prefix.json', 'w') as prefixfile:
        dump = {}
        for prefix, rankings in min_dist_per_prefix.items():
            dump[prefix] = []
            dump[prefix].append({
                'rankings': rankings
            })
        # dump json mappings out to dumpfile      
        json.dump(dump, prefixfile)

    end = time.time()
    print("time elapsed: {}".format(end-start)) 
    
if __name__ == "__main__":
    main()
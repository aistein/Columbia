#!/Users/alexstein/anaconda3/bin/python

# revtr_map_dests_to_dnet.py
# - parse all VP CSV files, one-by-one from the online source
# - filter out paths > 9 hops
# - write out a JSON file that maps each destination to its BGP-prefix
# - PURPOSE: the above file will be used to split data into test and training sets
# usage:
# - ./revtr_map_dests_to_dnet.py <username:password>
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
from collections import defaultdict

# for file download, etc.
import re, pycurl
from io import BytesIO
from bs4 import BeautifulSoup

# username and password for cget is passed via credentials.json file
try:
    credfile = "./credentials.json"
    with open(credfile, 'r') as jsonfile:
        json_data = json.load(jsonfile)
        userpwd = json_data["username"]+":"+json_data["password"]
except (FileNotFoundError,IndexError):
    raise ValueError('error: no/incorrect credentials.json file found.')

#==========================================================
# BGP Processing Functions
#==========================================================

# the BGP-dump and destinfo dict must be global
# - these will be accessed concurrently by multiple threads
# tag = datetime.datetime.now().strftime('%Y-%m-%d')
tag = "2018-11-29" # day of BGP data from when the original measurements were collected
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

# _prefixinfo
# spin off a _lookuptask(ip) thread
# returns tuple(thread-handle, ip-address)
def _prefixinfo(ip, destinfo):
    t = threading.Thread(target=_lookuptask(ip,destinfo), name=ip)
    t.start()
    return (t, ip)

# FilterAndCount ...
# Remove pings from the input CSV file for which the destination is never reached
# Yields ip-address
# Adapted from StackOverflow post: https://stackoverflow.com/a/17444799/3341596
def FilterAndCount(vp):
    if os.path.isfile("./vps" + vp + ".csv"):
        filename = "./vps" + vp + ".csv"
    else:
        filename = cget(vp+'.csv')
    with open(filename, "r") as csvfile:
        datareader = csv.reader(csvfile)
        # filter and count
        yield from map(lambda ping: ping[0],
            filter(lambda ping: _numhops(ping) <= 9, datareader))

    # these files are large, don't want to keep them around!
    # TODO: eventually uncomment and remove the files... need them now for data-splitting
    # os.remove(filename)
    return

# MapPrefixes ...
# Take filtered output and return the destination IP address; 
# Collect corresponding prefix info in the background
def MapPrefixes(vp,destinfo):
    dests = FilterAndCount(vp)
    # the below line will yeild a 2-tuple (thread_handle, IP address)
    yield from map(lambda dest: _prefixinfo(dest,destinfo), dests)
    return

# GroupByPrefix ...
# Group the (filtered) set of destinations by their BGP-routable prefix
# Note: this is the first function that forces an actual read! (doesn't "yeild")
dest_by_prefix = defaultdict(set)
def GroupByPrefix(vp):
    destinfo = {}
    for entry in MapPrefixes(vp, destinfo):
        # parse apart the returned entry
        thread = entry[0]    # thread handle
        dest = entry[1]      # destination ip-address

        # wait for the generated (updating destinfo) thread to finish
        thread.join()

        # gather the output
        if dest in destinfo:
            dinfo = destinfo[dest]
            asn = str(dinfo[0])     # ASN
            prefix = str(dinfo[1])  # BGP-routable prefix
        else:
            continue

        # DB lookup returned None, don't process
        if prefix == str(None) or asn == str(None):
            continue
        
        print("From VP:{}, mapping dest:{} to prefix {}.\n", vp, dest, prefix)

        # if the DB lookup succeeded, add "dest" to the map under "prefix"
        dest_by_prefix[prefix].add(dest)

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
    num_to_read = 10
    for hit in soup.find_all('a'):
        match = re.match(r'^.*csv', hit['href'])
        if match and num_to_read > 0:
            vplist.add(match[0])
            num_to_read = num_to_read - 1

    # done extracting, remove the vp-html-file
    os.remove(vplisthtml)

    for vpcsv in vplist:
        GroupByPrefix(os.path.splitext(vpcsv)[0]) # removes the ".csv"
        print("Processed " + vpcsv + "...")

    # dump into json file
    with open('mappings/dests_by_prefix.json', 'w') as prefixfile:    
        json.dump({ k:list(v) for k, v in dest_by_prefix.items() }, prefixfile)

    end = time.time()
    print("time elapsed: {}".format(end-start)) 
    
if __name__ == "__main__":
    main()
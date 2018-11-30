#!/Users/alexstein/anaconda3/bin/python

# usage:
# - ./revtr_query.py asn <asn number>
# - ./revtr_query.py prefix <prefix number>
# returns:
# - top 10 VP's ranked by distance to the queried prefix/asn

import sys, json
from operator import itemgetter

queryby = sys.argv[1]
if queryby == "asn":
    filename = 'mappings/min_by_asn.json'
elif queryby == "prefix":
    filename = 'mappings/min_by_prefix.json'
else:
    raise ValueError('incorrect query type - use one of (asn/prefix).')

querykey = sys.argv[2]

with open(filename, 'r') as jsonfile:
    json_data = json.load(jsonfile)
    try:
        cnt = 10
        for key, val in sorted(json_data[querykey][0]['rankings'].items(), key=lambda x: x[1]):
            print("VP: {: <40}, Dist: {}".format(key, val))
            cnt = cnt - 1
            if cnt == 0:
                break
    except KeyError:
        if queryby == "asn":
            print("ASN not found.")
        if queryby == "prefix":
            print("Prefix not found.")
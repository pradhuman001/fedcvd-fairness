import sys, json, analysis2 as B
scen=sys.argv[1]
res,n=B.cluster_bootstrap(scen,B=300)
json.dump({f"{k[0]}|{k[1]}":v for k,v in res.items()},open(f"res4/cluster_boot_{scen}.json","w")); print(scen,n)

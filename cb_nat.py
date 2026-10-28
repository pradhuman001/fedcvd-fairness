import analysis2 as B, json
res,n=B.cluster_bootstrap("natural",B=300)
json.dump({f"{k[0]}|{k[1]}":v for k,v in res.items()},open("res2/cluster_boot_natural.json","w"))
print("done",n,res)

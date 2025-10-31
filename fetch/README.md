```bash
pip install mygene

python map_to_ensembl.py \
  --genes genes.csv \
  --hgnc approved_symbols.txt \
  --output mapped.csv

```


```bash
awk -F"," 'FNR==NR{a[$1]=1; b[$1]=$3; next;}{gsub(/\r/, ""); if(a[$2] == 1){print $0","b[$2]} else { print $0",." }  }' mapped.final.csv mirtarbase.filtered.csv > mirtarbase.mapped.csv

awk -F"," '{ if(FNR==1){ print "mirna_id,gene_id,weight" } print $1","$3",1.0" }' mirtarbase.mapped.csv > gene_mirna.csv

awk -F"," '{ split($2, a, ";"); print $1","a[1]","$3 }' gene_mirna.csv > gene_mirna.final.csv
```
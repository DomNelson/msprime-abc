set -e

NINDS=$1
REC=$2
EFF=$3
MUT=$4

msp simulate $NINDS tmp --mutation-rate $MUT --effective-population-size $EFF --recombination-rate $REC && msp vcf --ploidy 2 tmp > tmp2 && mv tmp2 tmp

cat tmp | sed s/\_0/\_a/g > tmp2 && mv tmp2 tmp

plink2 --make-king square0 --vcf tmp #&& cat plink2.king

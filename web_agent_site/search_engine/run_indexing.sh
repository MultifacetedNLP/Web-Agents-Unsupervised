python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $SCRATCH/search_engine/resources_100 \
  --index $SCRATCH/search_engine/indexes_100 \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $SCRATCH/search_engine/resources \
  --index $SCRATCH/search_engine/indexes \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $SCRATCH/search_engine/resources_1k \
  --index $SCRATCH/search_engine/indexes_1k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input $SCRATCH/search_engine/resources_100k \
  --index $SCRATCH/search_engine/indexes_100k \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw

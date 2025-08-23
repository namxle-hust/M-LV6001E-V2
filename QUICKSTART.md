# from the project root (where Dockerfile.cpu lives)

```bash
docker build -f Dockerfile.cpu -t level1-gnn:cpu .
docker run --rm --it \
 -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/data:/app/data" \
 level1-gnn:cpu bash
```


visual-fs:  # feature selection
	export PYTHONPATH="/home/shyeon/workspace/ds/graph-study/src":$$PYTHONPATH
	xvfb-run python3 src/visual/visual.py

# run-shell:
# 	docker exec -it mongo-with-fastapi_mongo_1 bash

# logs:
# 	docker logs mongo-with-fastapi_mongo_1 
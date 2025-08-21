# DIAMM

# **Quick Start**

1. Install dependencies:

We recommend deploying PostgreSQL and Redis using the official images via Docker or Kubernetes. Considering the resource pool size, the recommended parameter configurations are as follows.

Image：

[docker.io/ankane/pgvector:v0.5.1](http://docker.io/ankane/pgvector:v0.5.1)

[docker.io/library/redis:7.2-alpine](http://docker.io/library/redis:7.2-alpine)

PGVector ConfigMap

```markdown
work_mem = 64MB
shared_buffers = 6GB
maintenance_work_mem = 512MB
```

Redis ConfigMap

```markdown
redis.conf
maxmemory 10GB

bind 0.0.0.0
protected-mode no
requirepass "123456"
```

2.Start process:

Master node:

**Task result show process：**/vLLM-k8s-operator/user_tasks$   show.py

**Embedding process：**/vLLM-k8s-operator/models/embedding main.py

**Fastapi process：**/vLLM-k8s-operator python -m program.fastapi_app

**QMSD algorithm process：** /vLLM-k8s-operator/deployment/deployment_design$ algorithm_proposed.py

**DIAMM multimodal warm-up process：**/vLLM-k8s-operator/deployment/service_drop/main.py  

**Weight integrity check process：**/vLLM-k8s-operator/check_weight$ main.py

**Prometheus-DGCM check process：**/vLLM-k8s-operator/check_weight/Prometheus_start.py

**Web start** web-end.py

NFS:

**Weight send process： /NFS_server**/send_weight#  main_drop.py

**File service process：/NFS_server**/show_url.py

Celry config

/etc/supervisor/conf.d/celery.conf

```markdown

[program:celery_worker]

command=/bin/bash -c "source ~/.bashrc && conda activate video_vector && celery -A program.celery_app worker --concurrency=40 -Q celery --loglevel=debug -E --without-mingle -n worker_%(process_num)02d@%%h"

directory=/vLLM-k8s-operator; 
user=root
environment=
    PYTHONPATH="/vLLM-k8s-operator",
    CELERY_CONFIG_MODULE="program.celery_app"
autostart=true
autorestart=true
startretries=3
numprocs=4                        
process_name=%(program_name)s_%(process_num)02d
stopwaitsecs=30
killasgroup=true
priority=1000
redirect_stderr=true
stdout_logfile=/vLLM-k8s-operator/log/celery/worker_%(process_num)02d.log
stdout_logfile_maxbytes=50MB
stdout_logfile_backups=10

[program:celery_flower]
command=/bin/bash -c "source ~/.bashrc && conda activate video_vector && export FLOWER_UNAUTHENTICATED_API=1 && celery -A program.celery_app flower --broker=redis://:123456@192.168.2.75:6379/0 --address=0.0.0.0 --port=9802"
    
directory=/vLLM-k8s-operator
user=root
autostart=true
autorestart=true
startretries=3
stopwaitsecs=30
redirect_stderr=true
stdout_logfile=/vLLM-k8s-operator/log/celery/flower.log

```

Celery start ：

sudo supervisorctl reread

sudo supervisorctl update

sudo supervisorctl restart all

# **Examples**

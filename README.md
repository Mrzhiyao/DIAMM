# DIAMM

# **Quick Start**

1. Install dependencies:

We recommend deploying PostgreSQL and Redis using the official images via Docker or Kubernetes. Considering the resource pool size, the recommended parameter configurations are as follows.

**PGVector Image**

[docker.io/ankane/pgvector:v0.5.1](http://docker.io/ankane/pgvector:v0.5.1)

**Redis Image**

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

**Multimodal models**

We deploy multimodal models on Kubernetes. To complete the test cases, your work nodes should ensure that each node has the following images available.

[docker.io/library/lmdeploy-new:v0.6.4-cu11](http://docker.io/library/lmdeploy-new:v0.6.4-cu11)

[docker.io/ollama/ollama:latest](http://docker.io/ollama/ollama:latest)

[docker.io/library/cogvideox2b:0.1](http://docker.io/library/cogvideox2b:0.1)

[docker.io/aoirint/sd_webui:0.2](http://docker.io/aoirint/sd_webui:0.2)

The Dockerfiles for the latter two images can be found in the Multimodal_files folder; they originate from the official model and an Xformer-optimized variant, respectively.

**Prometheus**

We query container status through Prometheus, which requires dgcm plugin to obtain GPU usage. The YAML file can be found in the folder /prometheus and needs to be loaded into the cluster.

**Runtime environment**

environment.yml

2.Start running process:

Master node:

```markdown
**Task result show process：
**/vLLM-k8s-operator/user_tasks$   show.py

**Embedding process：
**/vLLM-k8s-operator/models/embedding main.py

**Fastapi process：
**/vLLM-k8s-operator python -m program.fastapi_app

**QMSD algorithm process：
** /vLLM-k8s-operator/deployment/deployment_design$ algorithm_proposed.py

**DIAMM multimodal warm-up process：
**/vLLM-k8s-operator/deployment/service_drop/main.py  

**Weight integrity check process：
**/vLLM-k8s-operator/check_weight$ main.py

**Prometheus-DGCM check process：
**/vLLM-k8s-operator/check_weight/Prometheus_start.py

**Web start** 
web-end.py
```

NFS:

```markdown
**Weight send process： 
/NFS_server**/send_weight#  main_drop.py

**File service process：
/NFS_server**/show_url.py
```

**Celery start** 

Please run celery.conf in your path:

/etc/supervisor/conf.d/celery.conf

```markdown
sudo supervisorctl reread
sudo supervisorctl update
sudo supervisorctl restart all
```

Afterward, we will containerize the above processes and configure them with shell scripts. Once you start the above processes, you can use the DIAMM multimodal service in two ways:

1. **Concurrency Program Testing**
    
    Please visit `/test/send_task.py`, where you can control the concurrency, task types, and ratios.
    
2. **Web Access**
    
    We have forwarded the frontend through `web-end.py`, and you can access it directly at `localhost:8989`.
    

Note that if you find that executing tasks takes a long time (for infrequent use), you can save at least one copy of the commonly used models without clearing them.

# **Examples**

Celery-Flower ：Task execution status query

<img width="1216" height="908" alt="image" src="https://github.com/user-attachments/assets/26828f26-6aaf-41f0-96a0-25adcb56a9d9" />

 “`
 ![图片描述](./test/img/celery.jepg)
 “`

Web UI ：Task execution result

![github2](https://github.com/user-attachments/assets/5889cb0f-051e-4f7d-b538-72686a9462ba)


# **License**

This project is licensed under the MIT License. See the [LICENSE](https://github.com/NetX-lab/Ayo/blob/main/LICENSE) file for details.

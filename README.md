# campfire
Complementary Automated Methods for Proofreading

## Set up
1. Set up AWS credentials for IARPA MICrONS EXT AWS account 
2. Download `kubectl`, `eksctl`, and `awscli`
3. Update kubeconfig

```
aws eks --region region update-kubeconfig --name agents-cluster
```
4. Copy `~/.aws/credentials`, `~/.cloudvolume/secrets/cave-secret.json`, and `~/.neuvuequeue/neuvuequeue.cfg` to `/secrets`.
5. Build app docker image and push to ECR.


## General Steps

1. Populate queue with the channels you want to transfer.
2. Create a nodegroup using `eksctl` to add compute to your cluster.

```bash
eksctl create nodegroup --config-file=kube/{cpu|gpu}-nodegroup.yml 
```

3. If the app/ code changed, update the docker image on AWS ECR.
4. Verify that `kubectl` can connect to your cluster with `kubectl get nodes` and `kubectl get pods`.
5. Run the job with `kubectl apply -f kube/{agents|gpu}-deploy.yaml`


## Debug Steps

- To enter an interactive terminal for a pod

```bash
 kubectl exec -it [pod-name] -- /bin/bash/
```

- Delete a node group 

```bash
eksctl delete nodegroup --region=us-east-1 --cluster="spdb-to-cv-2" --name=spdb-to-cv-workers-2x
```

- Delete a job

```bash
kubectl delete jobs tips-jobs
```

- Add user to masters list 

```bash
eksctl create iamidentitymapping --cluster spdb-to-cv-2 --arn arn:aws:iam::407510763690:user/hiderrt1-bossdb --username hiderrt1-bossdb --group system:masters --region us-east-1
```

- Get masters list

```bash
eksctl get iamidentitymapping --cluster spdb-to-cv-2 --region us-east-1
```

- Get CPU/MEM usage

```
kubectl top nodes --tail
```

- Get logs for a pod 

```
kubectl logs {pod_name}
```
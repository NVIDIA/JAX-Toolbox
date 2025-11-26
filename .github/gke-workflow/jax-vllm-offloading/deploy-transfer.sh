kubectl apply -f deployment/gateway-pod.yml
kubectl apply -f deployment/gateway-svc.yml

kubectl apply -f huggingface-secret.yml

kubectl apply -f deployment/rollout.yml
kubectl apply -f deploymeny/trainer.yml

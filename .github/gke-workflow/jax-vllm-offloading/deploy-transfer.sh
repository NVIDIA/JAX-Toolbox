kubectl apply -f transfer/deployment/gateway-pod.yml
kubectl apply -f transfer/deployment/gateway-svc.yml

kubectl apply -f huggingface-secret.yml

kubectl apply -f transfer/deployment/rollout.yml
kubectl apply -f transfer/deployment/trainer.yml

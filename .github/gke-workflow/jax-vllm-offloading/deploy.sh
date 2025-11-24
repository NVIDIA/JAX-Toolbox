kubectl apply -f gateway-pod.yml
kubectl apply -f gateway-svc.yml

kubectl apply -f huggingface-secret.yml

kubectl apply -f rollout.yml
kubectl apply -f trainer.yml

#!/usr/bin/bash
minikube delete --all --purge
minikube config set memory 25000
minikube config set cpus 10
minikube start --kubernetes-version=1.22.17
kubectl create namespace train-ticket

cd train-ticket
make deploy Namespace=train-ticket
cd ..

#sleep 5m
#kubectl create -f prometheus-operator-crd
#kubectl apply -R -f monitoring


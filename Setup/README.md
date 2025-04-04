To access the grafana dashboard the following commands are required:

```
bash
kubectl -n monitoring port-forward svc/prometheus-operated 9090 # To access prometheus
kubectl -n grafana port-forward svc/grafana 3000
```

The username is admin and the password is devops123

The used dashboards in this work are found in the "Grafana Dashboard" Folder.

For more informations please consult the grafana [documentation](https://grafana.com/docs/)



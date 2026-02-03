# 1. Configurar alerta de billing (FAÇA ISSO PRIMEIRO!)
# Console AWS → Billing → Budgets → Create Budget
# Valor: $5.00 - Alerta em 50% e 80%
```

### Checklist Pós-Vídeo (IMPRIMA ISSO!)
```
□ 1. Deletar função Lambda
      Console → Lambda → lstm-vale3-predict → Delete

□ 2. Deletar imagem ECR
      Console → ECR → lstm-vale3-api → Delete repository

□ 3. Deletar API Gateway
      Console → API Gateway → Selecionar API → Delete

□ 4. Verificar CloudWatch Logs (acumula espaço)
      Console → CloudWatch → Log groups → Deletar logs do projeto

□ 5. CONFERIR NO DIA SEGUINTE
      Console → Billing → Verificar se zerou

# Remove imagens não usadas, containers parados, cache de build
docker system prune -a

## Verificar quanto espaço Docker está usando:
docker system df

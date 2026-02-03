# 🧹 CHECKLIST DE LIMPEZA AWS

## ⚠️ EXECUTE APÓS GRAVAR O VÍDEO!

---

## Ordem de Deleção (importante!)

### 1. ❑ Deletar API Gateway
```
Console AWS → API Gateway → lstm-vale3-api → Delete
```
Ou via CLI:
```bash
aws apigatewayv2 delete-api --api-id SEU_API_ID --region us-east-1
```

---

### 2. ❑ Deletar Função Lambda
```
Console AWS → Lambda → lstm-vale3-predict → Actions → Delete
```
Ou via CLI:
```bash
aws lambda delete-function --function-name lstm-vale3-predict --region us-east-1
```

---

### 3. ❑ Deletar Repositório ECR (imagem Docker)
```
Console AWS → ECR → lstm-vale3-api → Delete repository
```
⚠️ Marque "Delete all images" antes de confirmar!

Ou via CLI:
```bash
aws ecr delete-repository --repository-name lstm-vale3-api --force --region us-east-1
```

---

### 4. ❑ Deletar Role IAM
```
Console AWS → IAM → Roles → lambda-lstm-vale3-role → Delete
```
Ou via CLI:
```bash
# Primeiro remove a política
aws iam detach-role-policy --role-name lambda-lstm-vale3-role --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Depois deleta a role
aws iam delete-role --role-name lambda-lstm-vale3-role
```

---

### 5. ❑ Deletar Logs do CloudWatch
```
Console AWS → CloudWatch → Log groups → /aws/lambda/lstm-vale3-predict → Delete
```
Ou via CLI:
```bash
aws logs delete-log-group --log-group-name /aws/lambda/lstm-vale3-predict --region us-east-1
```

---

### 6. ❑ Deletar Alerta de Budget (opcional)
```
Console AWS → Billing → Budgets → tech-challenge-alert → Delete
```

---

## ✅ Verificação Final

### No dia seguinte, confira:

1. ❑ **Billing Dashboard**: https://console.aws.amazon.com/billing/
   - Custo deve estar zerado ou mínimo

2. ❑ **Resource Groups**: https://console.aws.amazon.com/resource-groups/
   - Procure por recursos com tag "lstm" ou "vale3"

3. ❑ **Cost Explorer**: https://console.aws.amazon.com/cost-management/
   - Verifique se não há cobranças novas

---

## 🚨 Se algo der errado

Se não conseguir deletar algum recurso:

1. Verifique dependências (alguns recursos dependem de outros)
2. Tente pelo Console AWS (às vezes é mais fácil que CLI)
3. Aguarde alguns minutos e tente novamente
4. Contate suporte AWS se persistir

---

## 📞 Suporte AWS

Se tiver cobranças inesperadas:
- AWS Support: https://console.aws.amazon.com/support/
- Pode solicitar reembolso em casos de uso acidental

---

**Data do Deploy:** ___/___/______

**Data da Limpeza:** ___/___/______

**Confirmação de Billing Zerado:** ❑ Sim

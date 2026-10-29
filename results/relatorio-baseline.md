# Relatório — Baseline (MLP)

Estado inicial do projeto, antes de qualquer melhoria de modelo. Serve de referência para comparar as próximas execuções (dados brutos em [treinos.csv](treinos.csv)).

## Configuração

| Item | Valor |
|---|---|
| Modelo | MLP `784 → 128 → 64 → 10` (`DigitClassifier.newBlock()`) |
| Dados | MNIST: 60.000 imagens de treino, 10.000 de validação (28×28, cinza) |
| Perda / métrica | `softmaxCrossEntropyLoss` / `Accuracy` |
| Otimizador, taxa de aprendizado | Padrões do DJL (não configurados) |
| Lote / épocas | 32 / 5 |
| Semente | 42 (fixa a inicialização dos pesos e o embaralhamento) |
| Engine | PyTorch 2.7.1 (build `cu128`) |
| Device | GPU 0 (RTX 5050 Laptop, 8 GB), forçada com `-Dtreino.device=gpu` |
| Máquina | i7-14650HX (24 threads), 14 GB RAM |

## Resultados (2 execuções, mesma semente)

| Execução | Acc. treino | Acc. validação | Perda treino | Perda validação | Tempo total | Imagens/s |
|---|---|---|---|---|---|---|
| baseline-mlp-r1 | 98,68% | **97,65%** | 0,0409 | 0,0802 | 24,8 s | 12.110 |
| baseline-mlp-r2 | 98,68% | **97,65%** | 0,0409 | 0,0802 | 23,1 s | 12.960 |

- **Reprodutível:** com a semente fixa, a acurácia e as perdas são idênticas entre execuções. Só o tempo varia (~7%).
- **Acurácia de referência: 97,65%** (2,35% de erro, ou 235 erros em 10.000 imagens de validação).

## Observações

- **Sobreajuste leve:** a acurácia de treino (98,68%) é cerca de 1 ponto acima da de validação (97,65%), e a perda de validação (0,080) é o dobro da de treino (0,041). A rede já memoriza um pouco o treino.
- **Antes da semente:** execuções sem semente variaram de 97,5% a 97,8% com a mesma configuração, ou seja, ~0,3 ponto de ruído. Diferenças menores que isso, sem semente fixa, não provam melhoria.
- **Desempenho:** o gargalo é a leitura dos PNGs do disco, não o cálculo. A GPU rende pouco com este modelo e lote (ver comparação CPU/GPU na conversa: ~1,6× com lote 32, ~1,1× com lote 512).
- **Limite de validade:** o conjunto de validação também é o de teste do MNIST e é usado para acompanhar o treino, então a nota é um pouco otimista.

## Como reproduzir

```bash
PYTORCH_FLAVOR=cu128 mvn exec:java -Dexec.mainClass=com.example.mnist.TrainMnist \
  -Dexec.args="32 5" -Dtreino.device=gpu -Dtreino.seed=42 -Dtreino.rotulo=baseline-mlp
```

Cada execução acrescenta uma linha a `results/treinos.csv`. Recompile com `mvn compile` depois de mudar o código (`exec:java` não recompila).

# Relatório comparativo — MLP (baseline) vs CNN

Mesmas condições nas duas: MNIST (60.000 treino / 10.000 validação), lote 32, 5 épocas, semente 42, otimizador padrão do DJL. Dados brutos em [treinos.csv](treinos.csv); baseline em [relatorio-baseline.md](relatorio-baseline.md).

## Arquiteturas

- **MLP:** `784 → 128 → 64 → 10`.
- **CNN:** `Conv 3x3 (32) → ReLU → Conv 3x3 (64) → ReLU → MaxPool 2x2 → Flatten → Linear 128 → ReLU → Linear 10`.

## Resultados

| Execução | Device | Acc. treino | **Acc. validação** | Perda validação | Tempo total | Imagens/s |
|---|---|---|---|---|---|---|
| MLP (r1 e r2, idênticas) | GPU | 98,68% | **97,65%** | 0,0802 | 23–25 s | 12.110–12.960 |
| CNN r1 | GPU | 99,64% | **98,77%** | 0,0490 | 28,7 s | 10.452 |
| CNN r2 | GPU | 99,67% | **98,64%** | 0,0588 | 28,9 s | 10.367 |
| CNN r1 | CPU | 99,69% | **98,52%** | 0,0575 | 173 s | 1.732 |
| CNN r2 | CPU | 99,69% | **98,52%** | 0,0575 | 204 s | 1.471 |

## Conclusões

- **Ganho de acurácia:** a CNN melhora de 97,65% para ~98,5–98,8% (+0,9 a +1,1 ponto; o erro cai de ~235 para ~125–150 imagens em 10.000, ou cerca de 40% menos erros). Isso é maior que o ruído de ~0,15 ponto medido entre execuções da CNN na GPU, então a melhoria é real.
- **Reprodutibilidade:**
  - MLP na GPU e CNN na CPU: resultados idênticos entre execuções com a mesma semente.
  - **CNN na GPU: não determinística** (98,77% vs 98,64%), provavelmente por algoritmos de convolução não determinísticos do cuDNN. Para comparações finas, use a CPU ou repita várias vezes e tire a média.
- **Custo:** na GPU a CNN custa ~20% mais tempo que o MLP (28,8 s contra ~24 s). Na CPU custa **7 a 8 vezes mais** que na GPU (173–204 s contra ~29 s), então com a CNN a GPU passa a fazer diferença real.
- **Sobreajuste:** a CNN chega a 99,6% no treino contra 98,7% na validação (gap ~0,9 ponto, parecido com o MLP). Há espaço para regularização (dropout, augmentation).
- **Ainda não medido:** desempenho em imagens fora do MNIST, que é o uso real. Ver próximos passos do projeto.

## Como reproduzir

```bash
mvn compile
PYTORCH_FLAVOR=cu128 mvn exec:java -Dexec.mainClass=com.example.mnist.TrainMnist \
  -Dexec.args="32 5" -Dtreino.device=gpu -Dtreino.seed=42 \
  -Dmodelo.arquitetura=cnn -Dtreino.rotulo=cnn
```

Os pesos são salvos em `build/mnist-model/mnist-cnn-0005.params`; o MLP continua em `mnist-mlp-0005.params`. Para classificar com a CNN, passe `-Dmodelo.arquitetura=cnn` (CLI e GUI usam MLP por padrão).

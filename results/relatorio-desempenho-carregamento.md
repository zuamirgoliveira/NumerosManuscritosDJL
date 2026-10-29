# Relatório — Otimização do carregamento de dados

GPU (RTX 5050), lote 32 (exceto onde indicado), 5 épocas, semente 42. Dados brutos em [treinos.csv](treinos.csv) (colunas `cache` e `threads`).

## O que foi feito

- **`-Dtreino.cache=true`:** `ImageFolderDataset` decodifica os 70.000 PNGs uma vez e guarda os pixels (uint8) na memória (~55 MB). Mesma conversão do caminho sem cache, então os tensores são idênticos.
- **`-Dtreino.threads=N`:** monta os lotes em N threads, em paralelo ao treino. Sem isso, o DJL monta os lotes na mesma thread do treino, em sequência.
- Ambos desligados por padrão (comportamento anterior preservado).

## Resultados

| Modelo | Configuração | Acc. validação | Tempo de fit | Imagens/s | Ganho vs. sem otimização |
|---|---|---|---|---|---|
| MLP | sem otimização | 97,65% | ~23–25 s (22,8 s na última medição) | 12.100–13.100 | — |
| MLP | cache | 97,65% | 15,1 s | 19.857 | ~1,5× |
| MLP | 8 threads | 97,65% | 11,6 s | — | ~2,0× |
| MLP | cache + 8 threads | 97,65% | **11,4 s** | 26.330 | **~2,0×** |
| CNN | sem otimização | 98,77% / 98,64% | 28,7–28,9 s | ~10.400 | — |
| CNN | cache | 98,87% | 20,4 s | 14.672 | ~1,4× |
| CNN | cache + 8 threads | 98,73% | **14,8 s** | 20.324 | **~1,9×** |
| CNN | cache, lote 128 | 98,80% | 13,9 s | 21.510 | ~2,1× |

## Conclusões

- **O gargalo era mesmo o carregamento.** Para o MLP, com a acurácia idêntica (97,65%) em todas as variantes, o treino ficou ~2× mais rápido; para a CNN, ~1,9×.
- **Threads rendem mais que o cache sozinho,** e as duas juntas são melhores. Para o MLP, as threads sozinhas (11,6 s) já atingem o que a combinação dá (11,4 s): o que restou é o custo do cálculo e de lançar kernels pequenos.
- **MLP continua com a GPU quase ociosa** (utilização média ~7–8%): a rede é pequena demais para ocupá-la. **A CNN ocupa mais** (média 28–38%, picos de 58–78%).
- A utilização média inclui a inicialização da JVM e do dataset, então subestima o valor durante o treino.
- **Reprodutibilidade:** MLP continua idêntico entre execuções. A CNN na GPU varia (98,64%–98,87%) por não determinismo da GPU, não pelas otimizações.

## Bug corrigido durante o teste

Um pool de threads comum (não daemon) impedia o programa de terminar: o treino acabava e gravava o resultado, mas o processo ficava preso. O pool agora usa threads daemon.

## Como usar

```bash
PYTORCH_FLAVOR=cu128 mvn exec:java -Dexec.mainClass=com.example.mnist.TrainMnist \
  -Dexec.args="32 5" -Dtreino.device=gpu -Dtreino.cache=true -Dtreino.threads=8
```

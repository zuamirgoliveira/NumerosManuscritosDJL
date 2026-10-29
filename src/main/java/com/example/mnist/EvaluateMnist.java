package com.example.mnist;

import ai.djl.MalformedModelException;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.Locale;
import java.util.stream.Stream;

/**
 * Avalia o modelo treinado sobre uma pasta inteira de imagens (inferência em
 * lote), no mesmo formato "uma subpasta numérica por classe" usado no treino.
 *
 * <p>Imprime a acurácia geral, a acurácia por dígito e a matriz de confusão
 * (linhas = dígito real, colunas = dígito previsto), que mostra QUAIS dígitos o
 * modelo confunde, algo que a acurácia geral esconde.
 *
 * <p>Uso: {@code EvaluateMnist [pasta]} (padrão: {@code images-validation}).
 * A arquitetura vem de {@code -Dmodelo.arquitetura=mlp|cnn}.
 */
public final class EvaluateMnist {

    private static final Path DEFAULT_ROOT = Path.of("images-validation");

    private EvaluateMnist() {}

    public static void main(String[] args)
            throws IOException, MalformedModelException, TranslateException {
        Path root = args.length > 0 ? Path.of(args[0]) : DEFAULT_ROOT;

        // confusion[real][previsto]
        int[][] confusion = new int[DigitClassifier.NUM_CLASSES][DigitClassifier.NUM_CLASSES];
        long start = System.nanoTime();

        try (DigitClassifier classifier = new DigitClassifier()) {
            for (int digit = 0; digit < DigitClassifier.NUM_CLASSES; digit++) {
                Path classDir = root.resolve(String.valueOf(digit));
                if (Files.notExists(classDir)) {
                    continue; // pasta de classe ausente: ignora (avaliação parcial)
                }
                for (Path image : listPngs(classDir)) {
                    int predicted = Integer.parseInt(classifier.classify(image).best().getClassName());
                    confusion[digit][predicted]++;
                }
            }
        }
        double seconds = (System.nanoTime() - start) / 1e9;
        printReport(root, confusion, seconds);
    }

    /** Lista os .png de uma pasta, em ordem. */
    private static List<Path> listPngs(Path dir) throws IOException {
        try (Stream<Path> files = Files.list(dir)) {
            return files.filter(f -> f.getFileName().toString().toLowerCase(Locale.ROOT).endsWith(".png"))
                    .sorted(Comparator.naturalOrder())
                    .toList();
        }
    }

    private static void printReport(Path root, int[][] confusion, double seconds) {
        long total = 0;
        long correct = 0;
        for (int real = 0; real < confusion.length; real++) {
            for (int predicted = 0; predicted < confusion[real].length; predicted++) {
                total += confusion[real][predicted];
                if (real == predicted) {
                    correct += confusion[real][predicted];
                }
            }
        }
        if (total == 0) {
            System.out.println("Nenhuma imagem .png encontrada em " + root.toAbsolutePath()
                    + ". Rode PrepareTrainingImages primeiro.");
            return;
        }

        System.out.printf("Pasta: %s | arquitetura: %s%n", root.toAbsolutePath(),
                DigitClassifier.Architecture.fromProperty());
        System.out.printf(Locale.ROOT, "Acurácia geral: %.2f%% (%d de %d) em %.1f s (%.0f imagens/s)%n",
                100.0 * correct / total, correct, total, seconds, total / seconds);

        System.out.println("\nAcurácia por dígito:");
        for (int digit = 0; digit < confusion.length; digit++) {
            int count = 0;
            for (int n : confusion[digit]) {
                count += n;
            }
            if (count > 0) {
                System.out.printf(Locale.ROOT, "  %d: %6.2f%% (%d imagens)%n",
                        digit, 100.0 * confusion[digit][digit] / count, count);
            }
        }

        System.out.println("\nMatriz de confusão (linha = real, coluna = previsto):");
        StringBuilder header = new StringBuilder("      ");
        for (int predicted = 0; predicted < confusion.length; predicted++) {
            header.append(String.format("%6d", predicted));
        }
        System.out.println(header);
        for (int real = 0; real < confusion.length; real++) {
            StringBuilder row = new StringBuilder(String.format("%4d |", real));
            for (int predicted = 0; predicted < confusion.length; predicted++) {
                row.append(String.format("%6d", confusion[real][predicted]));
            }
            System.out.println(row);
        }
    }
}

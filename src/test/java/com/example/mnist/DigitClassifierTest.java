package com.example.mnist;

import ai.djl.modality.Classifications;
import com.example.mnist.DigitClassifier.Architecture;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.EnumSource;

import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

/** Usa os modelos versionados em build/mnist-model (os testes rodam na raiz do projeto). */
class DigitClassifierTest {

    @ParameterizedTest
    @EnumSource(Architecture.class)
    void classificaImagemERetornaDezProbabilidadesQueSomamUm(Architecture architecture, @TempDir Path tmp)
            throws Exception {
        Path image = TestImages.write(tmp, "d.png", 0);
        try (DigitClassifier classifier = new DigitClassifier(DigitClassifier.DEFAULT_MODEL_DIR, architecture)) {
            Classifications result = classifier.classify(image);
            assertEquals(DigitClassifier.NUM_CLASSES, result.items().size());
            double sum = result.items().stream().mapToDouble(Classifications.Classification::getProbability).sum();
            assertEquals(1.0, sum, 1e-4);
        }
    }

    @Test
    void pastaSemModeloLancaIOException(@TempDir Path vazia) {
        assertThrows(IOException.class, () -> new DigitClassifier(vazia, Architecture.MLP));
    }

    @Test
    void avaliacaoEmLoteGeraRelatorioComAcuraciaEMatrizDeConfusao(@TempDir Path root) throws Exception {
        TestImages.write(root.resolve("0"), "a.png", 1);
        TestImages.write(root.resolve("7"), "a.png", 2);
        PrintStream original = System.out;
        ByteArrayOutputStream captured = new ByteArrayOutputStream();
        System.setOut(new PrintStream(captured, true, "UTF-8"));
        try {
            EvaluateMnist.main(new String[] {root.toString()});
        } finally {
            System.setOut(original);
        }
        String report = captured.toString("UTF-8");
        assertTrue(report.contains("Acurácia geral"), report);
        assertTrue(report.contains("de 2)"), report);
        assertTrue(report.contains("Matriz de confusão"), report);
    }
}

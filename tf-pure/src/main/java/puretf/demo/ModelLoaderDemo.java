package puretf.demo;

import puretf.data.IdxDataset;
import puretf.nn.Metrics;
import puretf.nn.Sequential;

import java.nio.file.Path;

/**
 * Loads a saved PTF1 model and evaluates it on Fashion-MNIST / MNIST test
 * files (no training). Also renders a per-class sample grid with predicted
 * labels and class accuracy.
 *
 * Usage: main [modelPath] [dataDir] [--show]
 *   modelPath default: target\fashion-cnn.bin
 *   dataDir   default: C:\Users\dell\Desktop\data\fashion
 */
public final class ModelLoaderDemo {

    private static final String[] NAMES = {
        "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        String modelPath = "target\\fashion-cnn.bin";
        String dir = "C:\\Users\\dell\\Desktop\\data\\fashion";
        boolean show = false;
        int batch = 512;
        for (String a : args) {
            if (a.equals("--show")) show = true;
            else if (a.startsWith("--batch=")) {
                batch = Integer.parseInt(a.substring("--batch=".length()));
                if (batch <= 0) throw new IllegalArgumentException(
                        "--batch must be positive");
            } else if (a.endsWith(".bin")) modelPath = a;
            else dir = a;
        }

        long t0 = System.currentTimeMillis();
        Sequential net = Sequential.loadFrom(Path.of(modelPath));
        System.out.printf("loaded model %s (%.1fs)%n", modelPath,
                (System.currentTimeMillis() - t0) / 1000.0);

        var teI = IdxDataset.loadImages(dir + "\\t10k-images-idx3-ubyte.gz");
        var teL = IdxDataset.loadLabels(dir + "\\t10k-labels-idx1-ubyte.gz");
        float[][] testX = teI.pixels();
        int[] testY = teL.labels();
        System.out.printf("loaded %d test %dx%d images (%.1fs)%n",
                testX.length, teI.rows(), teI.cols(),
                (System.currentTimeMillis() - t0) / 1000.0);

        System.out.printf("evaluating with batch size %d ...%n", batch);
        double[] eval = net.evaluate(testX, testY, 10, batch);
        int[][] conf = new int[10][10];
        float[][] logits = net.predict(testX, batch);
        int[] yPred = new int[testX.length];
        for (int i = 0; i < testX.length; i++) {
            yPred[i] = Metrics.argmax(logits[i]);
            conf[testY[i]][yPred[i]]++;
        }
        System.out.println();
        Metrics.printReport(eval[0], eval[1], conf, NAMES);

        int[] sampleIdx = new int[10];
        double[] accPct = new double[10];
        for (int i = 0; i < testX.length; i++) {
            int c = testY[i];
            if (sampleIdx[c] == 0 && yPred[i] == c) sampleIdx[c] = i;
        }
        for (int i = 0; i < 10; i++)
            if (sampleIdx[i] == 0) sampleIdx[i] = i;
        for (int c = 0; c < 10; c++) {
            int total = 0;
            for (int j = 0; j < 10; j++) total += conf[c][j];
            accPct[c] = total > 0 ? 100.0 * conf[c][c] / total : 0;
        }

        puretf.chart.EvalCharts.render(
                "Loaded model - one sample per class (true | pred | accuracy)",
                testX, teI.rows(), sampleIdx, testY, yPred, NAMES, accPct,
                show, "target\\fashion-loaded-eval.png");
        System.out.println("evaluation grid saved to target/fashion-loaded-eval.png"
                + (show ? " and opened in a window" : ""));
    }

    private ModelLoaderDemo() {}
}
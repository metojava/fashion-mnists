import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Result;
import org.tensorflow.Operand;
import org.tensorflow.op.Ops;
import org.tensorflow.op.core.Placeholder;
import org.tensorflow.op.core.Variable;
import org.tensorflow.types.TFloat32;
import org.tensorflow.types.TInt64;
import org.tensorflow.ndarray.Shape;
import org.tensorflow.ndarray.StdArrays;

import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.util.Arrays;

public class FashionMnistClassifierc {

    private static final String DATA_DIR = "data/fashion/";
    private static final int NUM_CLASSES = 10;
    
    private static final String[] LABELS = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        System.out.println("=========================================================");
        System.out.println("   TensorFlow Java Engine: Initializing Fashion MNIST   ");
        System.out.println("=========================================================");
        
        // 1. Load Data using our secondary helper file
        float[][][] trainImages = FashionMnistDataset.parseImages(DATA_DIR + "train-images-idx3-ubyte.gz");
        long[] trainLabels = FashionMnistDataset.parseLabels(DATA_DIR + "train-labels-idx1-ubyte.gz");
        float[][][] testImages = FashionMnistDataset.parseImages(DATA_DIR + "t10k-images-idx3-ubyte.gz");
        long[] testLabels = FashionMnistDataset.parseLabels(DATA_DIR + "t10k-labels-idx1-ubyte.gz");

        int numTestImages = testImages.length;

        // 2. Setup Network Architecture
        try (Graph g = new Graph(); Session session = new Session(g)) {
            Ops tf = Ops.create(g);

            Placeholder<TFloat32> inputPlaceholder = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, 28, 28)));
            Placeholder<TInt64> labelsPlaceholder = tf.placeholder(TInt64.class, Placeholder.shape(Shape.of(-1)));

            Operand<TFloat32> flatten = tf.reshape(inputPlaceholder, tf.constant(new long[]{-1, 784}));

            Variable<TFloat32> w1 = tf.variable(Shape.of(784, 128), TFloat32.class);
            Variable<TFloat32> b1 = tf.variable(Shape.of(128), TFloat32.class);
            Variable<TFloat32> w2 = tf.variable(Shape.of(128, NUM_CLASSES), TFloat32.class);
            Variable<TFloat32> b2 = tf.variable(Shape.of(NUM_CLASSES), TFloat32.class);

            session.runner().addTarget(tf.assign(w1, tf.random.statelessRandomNormal(tf.constant(new long[]{784, 128}), tf.constant(new long[]{1, 2})))).run();
            session.runner().addTarget(tf.assign(b1, tf.zeros(tf.constant(new long[]{128}), TFloat32.class))).run();
            session.runner().addTarget(tf.assign(w2, tf.random.statelessRandomNormal(tf.constant(new long[]{128, NUM_CLASSES}), tf.constant(new long[]{3, 4})))).run();
            session.runner().addTarget(tf.assign(b2, tf.zeros(tf.constant(new long[]{NUM_CLASSES}), TFloat32.class))).run();

            Operand<TFloat32> layer1 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, w1), b1));
            Operand<TFloat32> logits = tf.math.add(tf.linalg.matMul(layer1, w2), b2);
            Operand<TInt64> predictedLabels = tf.math.argMax(tf.nn.softmax(logits), tf.constant(1), TInt64.class);

            Operand<TFloat32> loss = tf.math.mean(tf.nn.softmaxCrossEntropyWithLogits(logits, tf.oneHot(labelsPlaceholder, tf.constant(NUM_CLASSES), tf.constant(1.0f), tf.constant(0.0f))).loss(), tf.constant(0));

            // Set up Backpropagation tree calculations
            org.tensorflow.op.core.Gradients gradients = org.tensorflow.op.core.Gradients.create(tf.scope(), loss, Arrays.asList(w1, b1, w2, b2));
            Operand<TFloat32> updateW1 = tf.assignSub(w1, tf.math.mul(gradients.dy(0), tf.constant(0.01f)));
            Operand<TFloat32> updateB1 = tf.assignSub(b1, tf.math.mul(gradients.dy(1), tf.constant(0.01f)));
            Operand<TFloat32> updateW2 = tf.assignSub(w2, tf.math.mul(gradients.dy(2), tf.constant(0.01f)));
            Operand<TFloat32> updateB2 = tf.assignSub(b2, tf.math.mul(gradients.dy(3), tf.constant(0.01f)));

            System.out.println("\nTraining over 5 Epochs...");
            try (TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(trainImages));
                 TInt64 labelTensor = TInt64.tensorOf(StdArrays.ndCopyOf(trainLabels))) {

                for (int epoch = 1; epoch <= 5; epoch++) {
                    long start = System.currentTimeMillis();
                    try (Result trainResult = session.runner()
                            .feed(inputPlaceholder.asOutput(), inputTensor)
                            .feed(labelsPlaceholder.asOutput(), labelTensor)
                            .fetch(loss)
                            .addTarget(updateW1).addTarget(updateB1).addTarget(updateW2).addTarget(updateB2)
                            .run()) {
                        TFloat32 calcLoss = (TFloat32) trainResult.get(0);
                        System.out.printf("Epoch %d/5 Finished | Loss: %.5f | Duration: %d ms%n", epoch, calcLoss.getFloat(), (System.currentTimeMillis() - start));
                    }
                }
            }

            // 3. Inference and Matrix Generation
            int[][] confusionMatrix = new int[NUM_CLASSES][NUM_CLASSES];
            long[] predictionsArray = new long[numTestImages];

            try (TFloat32 testInputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(testImages))) {
                try (Result evalResult = session.runner().feed(inputPlaceholder.asOutput(), testInputTensor).fetch(predictedLabels).run()) {
                    TInt64 testPredictions = (TInt64) evalResult.get(0);
                    for (int i = 0; i < numTestImages; i++) {
                        int actualClass = (int) testLabels[i];
                        int predictedClass = (int) testPredictions.getLong(i);
                        predictionsArray[i] = predictedClass;
                        confusionMatrix[actualClass][predictedClass]++;
                    }
                }
            }

            // Render Results Output elements
            printConfusionMatrix(confusionMatrix);
            int corrects = 0;
            for (int i = 0; i < NUM_CLASSES; i++) corrects += confusionMatrix[i][i];
            double finalAccuracy = ((double) corrects / numTestImages) * 100.0;

            displayVisualResults(testImages, testLabels, predictionsArray, finalAccuracy);
        }
    }

    private static void displayVisualResults(float[][][] images, long[] actual, long[] predicted, double accuracy) {
        SwingUtilities.invokeLater(() -> {
            JFrame frame = new JFrame("Fashion MNIST Test Evaluation Grid");
            frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
            frame.setSize(900, 800);
            frame.setLayout(new BorderLayout());

            JPanel header = new JPanel();
            header.setBackground(new Color(40, 44, 52));
            JLabel title = new JLabel(String.format("Classification Accuracy Matrix Result: %.2f%%", accuracy));
            title.setForeground(Color.WHITE);
            title.setFont(new Font("SansSerif", Font.BOLD, 16));
            header.add(title);
            frame.add(header, BorderLayout.NORTH);

            JPanel grid = new JPanel(new GridLayout(5, 5, 8, 8));
            grid.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

            for (int i = 0; i < 25; i++) {
                JPanel card = new JPanel(new BorderLayout());
                card.setBorder(BorderFactory.createLineBorder(Color.GRAY, 1));

                BufferedImage img = new BufferedImage(28, 28, BufferedImage.TYPE_BYTE_GRAY);
                for (int r = 0; r < 28; r++) {
                    for (int c = 0; c < 28; c++) {
                        int val = (int) (images[i][r][c] * 255);
                        img.setRGB(c, r, (val << 16) | (val << 8) | val);
                    }
                }

                JLabel imgLabel = new JLabel(new ImageIcon(img.getScaledInstance(90, 90, Image.SCALE_FAST)));
                card.add(imgLabel, BorderLayout.CENTER);

                String text = String.format("<html>Pred: %s<br>True: %s</html>", LABELS[(int)predicted[i]], LABELS[(int)actual[i]]);
                JLabel info = new JLabel(text, JLabel.CENTER);
                info.setOpaque(true);
                info.setFont(new Font("SansSerif", Font.PLAIN, 11));
                
                if (predicted[i] == actual[i]) {
                    info.setBackground(new Color(230, 245, 230));
                    info.setForeground(new Color(0, 100, 0));
                } else {
                    info.setBackground(new Color(255, 230, 230));
                    info.setForeground(new Color(150, 0, 0));
                }

                card.add(info, BorderLayout.SOUTH);
                grid.add(card);
            }
            frame.add(new JScrollPane(grid), BorderLayout.CENTER);
            frame.setLocationRelativeTo(null);
            frame.setVisible(true);
        });
    }

    private static void printConfusionMatrix(int[][] cm) {
        System.out.println("\n=== Confusion Matrix ===");
        System.out.printf("%-15s", "Actual \\ Pred");
        for (int i = 0; i < NUM_CLASSES; i++) System.out.printf(" [%d]  ", i);
        System.out.println();
        for (int r = 0; r < NUM_CLASSES; r++) {
            System.out.printf("%-15s", String.format("[%d] %s", r, LABELS[r]));
            for (int c = 0; c < NUM_CLASSES; c++) System.out.printf("%5d ", cm[r][c]);
            System.out.println();
        }
    }
}

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
import java.util.Arrays;

public class FashionMnistClassifiere {

    private static final String DATA_DIR = "data/fashion/";
    private static final int NUM_CLASSES = 10;
    private static final int BATCH_SIZE = 64;
    private static final float LEARNING_RATE = 0.005f;
    
    private static final String[] LABELS = {
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", 
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    };

    public static void main(String[] args) throws Exception {
        System.out.println("=========================================================");
        System.out.println("   TensorFlow Java Engine: Initializing Fashion MNIST   ");
        System.out.println("=========================================================");
        
        float[][][] trainImages = FashionMnistDataset.parseImages(DATA_DIR + "train-images-idx3-ubyte.gz");
        long[] trainLabels = FashionMnistDataset.parseLabels(DATA_DIR + "train-labels-idx1-ubyte.gz");
        float[][][] testImages = FashionMnistDataset.parseImages(DATA_DIR + "t10k-images-idx3-ubyte.gz");
        long[] testLabels = FashionMnistDataset.parseLabels(DATA_DIR + "t10k-labels-idx1-ubyte.gz");

        int numTrain = trainImages.length;
        int numTestImages = testImages.length;

        try (Graph g = new Graph(); Session session = new Session(g)) {
            Ops tf = Ops.create(g);

            Placeholder<TFloat32> inputPlaceholder = tf.placeholder(TFloat32.class, Placeholder.shape(Shape.of(-1, 28, 28)));
            Placeholder<TInt64> labelsPlaceholder = tf.placeholder(TInt64.class, Placeholder.shape(Shape.of(-1)));

            Operand<TFloat32> flatten = tf.reshape(inputPlaceholder, tf.constant(new long[]{-1, 784}));

            Variable<TFloat32> w1 = tf.variable(Shape.of(784, 128), TFloat32.class);
            Variable<TFloat32> b1 = tf.variable(Shape.of(128), TFloat32.class);
            Variable<TFloat32> w2 = tf.variable(Shape.of(128, NUM_CLASSES), TFloat32.class);
            Variable<TFloat32> b2 = tf.variable(Shape.of(NUM_CLASSES), TFloat32.class);

            float scale1 = (float) Math.sqrt(2.0 / 784.0);
            float scale2 = (float) Math.sqrt(2.0 / 128.0);

            Operand<TFloat32> randW1 = tf.math.mul(tf.random.statelessRandomNormal(tf.constant(new long[]{784, 128}), tf.constant(new long[]{1, 2})), tf.constant(scale1));
            Operand<TFloat32> randW2 = tf.math.mul(tf.random.statelessRandomNormal(tf.constant(new long[]{128, NUM_CLASSES}), tf.constant(new long[]{3, 4})), tf.constant(scale2));

            session.runner().addTarget(tf.assign(w1, randW1)).run();
            session.runner().addTarget(tf.assign(b1, tf.zeros(tf.constant(new long[]{128}), TFloat32.class))).run();
            session.runner().addTarget(tf.assign(w2, randW2)).run();
            session.runner().addTarget(tf.assign(b2, tf.zeros(tf.constant(new long[]{NUM_CLASSES}), TFloat32.class))).run();

            Operand<TFloat32> layer1 = tf.nn.relu(tf.math.add(tf.linalg.matMul(flatten, w1), b1));
            Operand<TFloat32> logits = tf.math.add(tf.linalg.matMul(layer1, w2), b2);
            Operand<TInt64> predictedLabels = tf.math.argMax(tf.nn.softmax(logits), tf.constant(1), TInt64.class);

            Operand<TFloat32> loss = tf.math.mean(tf.nn.softmaxCrossEntropyWithLogits(logits, tf.oneHot(labelsPlaceholder, tf.constant(NUM_CLASSES), tf.constant(1.0f), tf.constant(0.0f))).loss(), tf.constant(0));

            org.tensorflow.op.core.Gradients gradients = org.tensorflow.op.core.Gradients.create(tf.scope(), loss, Arrays.asList(w1, b1, w2, b2));
            Operand<TFloat32> updateW1 = tf.assignSub(w1, tf.math.mul(gradients.dy(0), tf.constant(LEARNING_RATE)));
            Operand<TFloat32> updateB1 = tf.assignSub(b1, tf.math.mul(gradients.dy(1), tf.constant(LEARNING_RATE)));
            Operand<TFloat32> updateW2 = tf.assignSub(w2, tf.math.mul(gradients.dy(2), tf.constant(LEARNING_RATE)));
            Operand<TFloat32> updateB2 = tf.assignSub(b2, tf.math.mul(gradients.dy(3), tf.constant(LEARNING_RATE)));

            System.out.println("\nTraining over 5 Epochs (using Mini-Batch Optimization)...");
            for (int epoch = 1; epoch <= 5; epoch++) {
                long start = System.currentTimeMillis();
                float totalLoss = 0;
                int batches = 0;

                for (int i = 0; i < numTrain; i += BATCH_SIZE) {
                    int currentBatchSize = Math.min(BATCH_SIZE, numTrain - i);
                    
                    float[][][] batchImg = FashionMnistDataset.getBatchImages(trainImages, i, currentBatchSize);
                    long[] batchLab = FashionMnistDataset.getBatchLabels(trainLabels, i, currentBatchSize);

                    try (TFloat32 inputTensor = TFloat32.tensorOf(StdArrays.ndCopyOf(batchImg));
                         TInt64 labelTensor = TInt64.tensorOf(StdArrays.ndCopyOf(batchLab))) {

                        try (Result trainResult = session.runner()
                                .feed(inputPlaceholder.asOutput(), inputTensor)
                                .feed(labelsPlaceholder.asOutput(), labelTensor)
                                .fetch(loss)
                                .addTarget(updateW1).addTarget(updateB1).addTarget(updateW2).addTarget(updateB2)
                                .run()) {
                            TFloat32 calcLoss = (TFloat32) trainResult.get(0);
                            totalLoss += calcLoss.getFloat();
                            batches++;
                        }
                    }
                }
                System.out.printf("Epoch %d/5 Finished | Avg Loss: %.5f | Duration: %d ms%n", epoch, (totalLoss / batches), (System.currentTimeMillis() - start));
            }

            int[][] confusionMatrix = new int[NUM_CLASSES][NUM_CLASSES];
            long[] predictionsArray = new long[numTestImages];

            System.out.println("\nRunning full evaluation over test set...");
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

            printConfusionMatrix(confusionMatrix);
            int corrects = 0;
            for (int i = 0; i < NUM_CLASSES; i++) corrects += confusionMatrix[i][i];
            double finalAccuracy = ((double) corrects / numTestImages) * 100.0;

            // Trigger separated visual view matrix component
            FashionMnistVisualizer.displayVisualResults(testImages, testLabels, predictionsArray, finalAccuracy);
        }
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

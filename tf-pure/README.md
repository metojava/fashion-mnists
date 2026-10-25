# pure-tf-java

A **pure-Java** define-by-run autograd engine and Keras-flavoured neural
network library - zero TensorFlow, zero JNI, zero native code, zero
dependencies. One small jar carries the forward math, reverse-mode autodiff,
optimizers, training loop and model persistence.

This is the native-free replacement for tensorflow-capi-java: every layer
that used to emit TF graph ops now runs on the built-in tape with hand-written
analytic gradients.

there are branches : main and master in this repo 

first clone :

git clone https://github.com/metojava/pure-tf-java.git

cd pure-tf-java

git checkout main

or 

git checkout master

master has more content


## Quick start

```
run.cmd                            # XOR + regression demos
run.cmd gradcheck                  # finite-difference verification of every op
run.cmd features                   # integration smoke test (MLP / attention / LSTM / CNN)
run.cmd fashion [5]                # Fashion-MNIST MLP (784-256-10)
run.cmd fashion-cnn [5] [--fast] [--batch=N]
                                   # Fashion-MNIST model, --fast = 12k subset,
                                   # --batch = train/eval batch size (default 512)
run.cmd load-model [--show] [--batch=N]
                                   # load target\fashion-cnn.bin and evaluate on
                                   # the 10k test set; --show opens the sample grid
run.cmd full [5] [--fast] [--show] [--batch=N]
                                   # train (fashion-cnn) then evaluate the saved model
```

Verified on this machine (JDK 23, Ryzen 16-core):

```
run.cmd fashion-cnn 1 --fast --batch=256   # trains CNN, 7.0s/epoch
run.cmd full 1 --fast --batch=256          # trains then evals with correct batch
```

Supported flags across the demos:

| Flag | Effect |
|---|---|
| `--fast` | train on a 12k/2k subset for a fast first result |
| `--batch=N` | set the mini-batch size for training **and** inference |
| `--show` | open the per-class prediction grid in a Swing window |

Requirements: JDK **21+** (JDK 23 recommended), Maven. Nothing else.

The vectorized `matmul` uses the HotSpot Vector API (`jdk.incubator.vector`).
`run.cmd` handles it automatically: it selects `C:\Program Files\Java\jdk-23`
when present and exports `MAVEN_OPTS=-Xmx4g --add-modules jdk.incubator.vector`
(same JVM as `exec:java`, so JVM flags must come from `MAVEN_OPTS`, not
`exec-maven-plugin` `<arguments>` â€” those are program args to `main()`).

To run `mvn exec:java` directly (equivalent of `run.cmd fashion-cnn 1 --fast --batch=256`):

```
set JAVA_HOME=C:\Program Files\Java\jdk-23
set MAVEN_OPTS=-Xmx4g --add-modules jdk.incubator.vector
mvn -q compile "-Dexec.mainClass=puretf.demo.CnnFashionMnistDemo" exec:java "-Dexec.args=C:\Users\dell\Desktop\data\fashion 1 --fast --batch=256"
```

## Evaluation & image grids

`evalCharts` writes a grid with one sample per class: the image, true label,
predicted label and the class-accuracy percentage under each image.
`ModelLoaderDemo` loads a saved `PTF1` model (default `target\fashion-cnn.bin`)
and reports loss, accuracy, per-class precision/recall/F1, confusion matrix
and the grid (`target\fashion-loaded-eval.png`).

Inference is batched internally (`Sequential.predict(X, batch)`, default 512)
so the autograd tape never holds the whole test set at once; the MLP used to
edge against the `-Xmx4g` limit. `evaluate(X, Y, classes, batch)` mirrors it.

## Layer zoo

| Group | Layers |
|---|---|
| Core | Dense, Dropout, Flatten, Reshape, Sequential |
| Activations | ReLU, LeakyReLU, ELU, SELU, SiLU, GELU, Softplus, Sigmoid, Tanh, Softmax |
| Normalization | LayerNorm1D, BatchNorm1D, BatchNorm2D |
| Vision | Conv2D (fused im2col-free), MaxPool2D, AvgPool2D, ZeroPad2D, Upsample2D |
| Sequence | Embedding, PositionalEncoding, SelfAttention, MultiHeadAttention (causal opt.), RNN, LSTM, GRU |

Every layer's gradient is verified numerically by `run.cmd gradcheck`
(30 checks, central differences).

## Keras-style trainer + persistence

```java
Sequential net = new Sequential()
        .add(new Conv2D(8, 3, 3, 1, 1, Padding.SAME)).add(new ReLU())
        .add(new MaxPool2D(2, 2, 2, 2, Padding.VALID))
        .add(new Flatten())
        .add(new Dense(128, Dense.Act.RELU))
        .add(new Dense(10));

History h = net.fitHistory(trainX, labels, 10,
        30 /*epochs*/, 96 /*batch*/,
        new Optim.Adam(0.0015f),
        LrSchedule.cosine(0.0015f, 0.0002f, 30),
        0.1f,                                    // validation split
        new EarlyStopping(Monitor.LOSS, 6, 0),   // patience 6
        true);
// Epoch k/N - loss .. - accuracy .. - val_loss .. - val_acc ..

net.saveTo(Path.of("model.bin"));                 // PTF1 binary format
Sequential restored = Sequential.loadFrom(Path.of("model.bin"));
TrainingCurves.save("curves", h, "target/curves.png");   // loss/acc PNG
```

`Optim.Stepper` implementations: `Adam`, `Momentum`, `RMSProp` (+ static
`sgd`). `LrSchedule`: `constant`, `step`, `cosine`, `exponential`, or a lambda.

## Low-level tape API

The layers compile down to the raw autograd tape, which remains fully
accessible:

```java
Tensor w1 = Tensor.glorot(rnd, 2, 8, 2, 8);      // trainable param
Tensor b1 = Tensor.param(new float[8], 8);

for (int step = 0; step < steps; step++) {
    Tensor.resetGraph();                          // fresh tape
    var x  = Tensor.of(batchX, B, 2);             // input leaf
    var loss = x.matmul(w1).addBias(b1).tanh().matmul(w2).mse(batchY);
    for (Tensor p : params) p.zeroGrad();
    loss.backward();                              // full backprop
    adam.step(params);
}
```

### Ops (each with analytic gradients)

- Linear algebra: `matmul`, `bmm` (batched), `addBias`, broadcast `broadcastAdd` /
  `broadcastMul` (NumPy right-aligned rules)
- Shape: `reshape`, `permute`, `sliceLast`, `rows`, `concatLast`,
  `meanAxis` / `sumAxis`
- Elementwise: `mul`, `scale`, `addScalar`, `power`
- Activations: `relu`, `leakyRelu`, `elu`, `selu`, `silu`, `gelu`, `softplus`,
  `sigmoid`, `tanh`, row-`softmax`
- Gather/vision: `embedding`, `pad2d`, `im2col` (col2im backward),
  `maxPool2d` (argmax routing), `avgPool2d`, `upsample2d`
- Losses: `mse`, fused `softmaxCE`

## Architecture notes

- Tape-based define-by-run autograd (micrograd lineage): each op records a
  backprop closure; `backward()` walks creation order in reverse.
- Conv2D is im2col + the tested rank-2 matmul, so its correctness rides on
  already-verified primitives; pooling ops route gradients via argmax masks.
- RNN/LSTM/GRU unroll through primitive ops - BPTT falls out of the tape for
  free.
- Model format `PTF1`: layer type codes + spec args + raw parameter tensors +
  extra state (BatchNorm running stats).
- Max-pool is piecewise linear; its gradient check uses well-separated window
  maxima (finite differences are meaningless at ties).
- `predict`/`evaluate` slice the dataset into batches so big test sets never
  materialize as one giant forward pass / tape.
- `matmul` (forward + `dA`/`dW` backward) is vectorized with the JDK Vector API
  (`FloatVector.SPECIES_PREFERRED`, scalar fallback when `n` is below the lane
  width). On JDK 23 this cut the Fashion-MNIST epoch from ~3.8 s to ~2.5 s at
  batch 512 with bit-identical losses.

## Parity status vs tensorflow-capi-java

Done: optimizers (SGD/Momentum/RMSProp/Adam), all activation layers, norm
layers, embedding + positional encoding, attention stack, recurrent stack
(RNN/LSTM/GRU), conv/pool/pad/upsample vision stack, early stopping, LR
schedules, validation split, model save/load, evaluation reports.

Not ported (by design): CrossAttention layer (needs two-input flows),
DepthwiseConv2D, SavedModel import/export (no native runtime to talk to),
GPU execution.

Performance expectation: showcase-sized models train comfortably on CPU
(Fashion-MNIST MLP ~2.5 s/epoch with the vectorized matmul at batch 512 on an
8-thread JDK 23; the `--fast` CNN subset trains in a similar range). Tried-and-
measured: register-tiled GEMM variants (k-blocked and row-blocked micro-kernels
holding the out row in registers) were slightly *slower* here - with n = 128 and
W L2-resident, the extra W re-reads and long per-accumulator FMA chains outweigh
the saved out round-trips, so the simple k-sequential streaming loop wins.


# Fashion-MNIST datasets

Run commands from the tfj folder:

```powershell
mvn compile exec:java
```

All three classifiers automatically download missing datasets into `data/fashion/`.
Existing nonempty files are reused. Both dataset readers also download missing files.

Download only, without TensorFlow training:

```powershell
mvn compile exec:java "-Dexec.mainClass=FashionMnistDownloader"
```

Source: https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/

Downloads use timeouts and temporary files, validate gzip and IDX magic headers,
and remove incomplete temporary files on failure.

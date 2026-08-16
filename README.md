# Barnes-Hut-t-SNE

A Java implementation of Barnes-Hut t-SNE, imported from [T-SNE-Java](https://github.com/lejon/T-SNE-Java)
(the reference implementation) and since worked over for performance. What was
changed, what it is worth, and what a caller has to know before upgrading is in
[Optimizations.md](Optimizations.md).

**It has no dependencies.** Everything the original took from third party libraries has been replaced
by code in this repository. The only entry in the POM is JUnit, at test scope.

## The MNIST benchmark

`MnistBenchmark` runs one full t-SNE over the MNIST training set. It is the measurement the numbers
below come from, and it is meant to be re-run rather than believed.

### What you need

Nothing beyond the repository. The data is in `src/test/resources/datasets`:

| file | what |
| --- | --- |
| `train-images-idx3-ubyte.gz` | the 60 000 training images, in the original IDX form |
| `train-labels-idx1-ubyte.gz` | their labels |
| `mnist2500_X.txt` | the 2500 row subset the demo uses |

`MnistData` reads the IDX files directly. It uses the `LEGACY` encoding, which reproduces
`mnist2500_X.txt` exactly, so a run over 60 000 is the same preprocessing as a run over the 2500 -
`MnistLegacyEncodingTest` is what keeps that true.

### Building and running

Everything below was run with **JDK 1.8.0_482, 64 bit**, from the repository root. The project
compiles to Java 8; a newer JDK will work, but the timings reported here are from 8.

PowerShell:

```powershell
$JDK = "F:\Java\jdk1.8.0_482-64bit"

New-Item -ItemType Directory -Force target\bench | Out-Null
$srcs = (Get-ChildItem -Recurse src\main\java -Filter *.java | ForEach-Object FullName) +
        "src\test\java\com\jujutsu\utils\MnistData.java" +
        "src\test\java\com\jujutsu\tsne\barneshut\MnistBenchmark.java"
& "$JDK\bin\javac.exe" -nowarn -d target\bench $srcs

& "$JDK\bin\java.exe" -Xmx10g -cp target\bench com.jujutsu.tsne.barneshut.MnistBenchmark
```

bash, same thing:

```bash
JDK=/f/Java/jdk1.8.0_482-64bit
mkdir -p target/bench
"$JDK/bin/javac" -nowarn -d target/bench $(find src/main/java -name '*.java') \
    src/test/java/com/jujutsu/utils/MnistData.java \
    src/test/java/com/jujutsu/tsne/barneshut/MnistBenchmark.java
"$JDK/bin/java" -Xmx10g -cp target/bench com.jujutsu.tsne.barneshut.MnistBenchmark
```

`MnistBenchmark` takes two optional arguments, the number of images and the number of iterations, and
defaults to **60 000 and 1000**. Everything else is fixed at what the demo uses, scaled up: two output
dimensions, the input reduced to 55 by PCA, perplexity 20, `theta = 0.5`, `ParallelBHTsne`.

```powershell
# a short run to check the setup works - a few seconds
& "$JDK\bin\java.exe" -Xmx10g -cp target\bench com.jujutsu.tsne.barneshut.MnistBenchmark 5000 200
```

**Keep the iteration count at 1000 or more if you want to compare against anything.** Below 1000 the
end of the early exaggeration moves - it is `min(250, maxIter / 4)` - and short runs are therefore not
comparable with the reference implementation, which fixes it at 250.

### Memory

`-Xmx10g` is what the numbers below were measured with, on a machine with 16 GB. The 60 000 images are
a `double[60000][784]`, which is 376 MB, `BHTSne` flattens them into another 376 MB before it starts,
and the PCA works on a centered copy on top of that. Smaller heaps have not been tried at this size;
`-Xmx4g` is enough for `5000` images.

### Reading the output

`tsne()` prints progress of its own; the number to compare is the total the benchmark prints at the
end:

```
--- MnistBenchmark: n = 60000, 1000 iterations ---
reading the data        :     0,51 s
tsne()                  :   302,85 s   <- the number to compare
```

The sum of squares of the embedding is printed as well. It is there to make a run that went quietly
wrong visible, not as a correctness check.

Expect roughly **3.5 to 4 minutes** for the default run on hardware like the one below.

## End to end, against the state before the optimization

Baseline is commit `7444506`, the last one before this work started. Four runs alternating
`baseline / current / current / baseline`, 60 000 images, 1000 iterations, `-Xmx10g`, JDK 1.8.0_482,
on a Ryzen 5 5600H with 12 logical cores and 16 GB.

| | baseline | current | |
| --- | ---: | ---: | ---: |
| `tsne()`, 60 000 images, 1000 iterations | 316.1 s | 228.7 s | **1.38x** |

Means of two runs each; the individual totals were 302.85 and 329.44 against 235.54 and 221.94.

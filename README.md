# IntelGemm
GEneral Matrix Multiplication with Intel Compiler and his powerfull Autoparallelization and Autovectorization


```
void MatMul0(int m, int n, int k, float* a, float* b, float* c) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			for (int k0 = 0; k0 < k; ++k0) {
				c[i * n + j] += a[i * k + k0] * b[k0 * n + j];
			}
		}
	}
}

void MatMul1(int m, int n, int k, float* a, float* b, float* c) {
	for (int i = 0; i < m; ++i) {
		for (int k0 = 0; k0 < k; ++k0) {
			for (int j = 0; j < n; ++j) {
				c[i * n + j] += a[i * k + k0] * b[k0 * n + j];
			}
		}
	}
}

void MatMul2(int m, int n, int k, float* a, float* b, float* c) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}
```

### The Output
```
HelloWorld Intel MKL Gemm
Matrix[2 3] A
     0    -4     4
     7     2    -4

Matrix[3 4] B
    -3    -2    -3    -2
    -3     4    -9     3
   -10    -5     8    -3

Matrix[2 4] A x B meth0
   -28   -36    68   -24
    13    14   -71     4

Matrix[2 4] A x B meth1
   -28   -36    68   -24
    13    14   -71     4

Matrix[2 4] A x B meth2
   -28   -36    68   -24
    13    14   -71     4

Start Bench M=1920 N=2560 and K=1280
Bench method 0
TIME    97.63 ms
TIME    95.18 ms
TIME    91.10 ms
TIME    89.63 ms
TIME    88.26 ms
Bench method 1
TIME    88.22 ms
TIME    88.36 ms
TIME    89.07 ms
TIME    86.03 ms
TIME    88.58 ms
Bench method 2
TIME    89.54 ms
TIME    92.78 ms
TIME    91.48 ms
TIME    91.43 ms
TIME    92.41 ms
End.
```

```
Start Bench M=3840 N=5120 and K=2560
Bench method 0
TIME   675.68 ms
TIME   668.07 ms
TIME   674.81 ms
TIME   677.28 ms
TIME   673.65 ms
Bench method 1
TIME   669.31 ms
TIME   666.31 ms
TIME   673.02 ms
TIME   665.42 ms
TIME   696.23 ms
Bench method 2
TIME   704.99 ms
TIME   688.36 ms
TIME   675.51 ms
TIME   676.15 ms
TIME   677.55 ms
End.
```

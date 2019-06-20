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
Matrix (2 3) A 
     5     4     0
    -3     6     1

Matrix (3 4) B 
     2     8    -4     1
     0     5     9     9
     8    -6    -9    -7

Matrix (2 4) A x B meth0 
    10    60    16    41
     2     0    57    44

Matrix (2 4) A x B meth1 
    10    60    16    41
     2     0    57    44

Matrix (2 4) A x B meth2 
    10    60    16    41
     2     0    57    44

Start Bench M=1920 N=2560 and K=1280 
Bench methode 0
TIME    83.53 ms 
TIME    90.73 ms 
TIME    90.53 ms 
TIME    92.97 ms 
TIME    89.62 ms 
Bench methode 1
TIME    89.26 ms 
TIME    90.08 ms 
TIME    92.57 ms 
TIME    88.27 ms 
TIME    92.38 ms 
Bench methode 2
TIME   104.02 ms 
TIME    92.48 ms 
TIME    90.05 ms 
TIME    91.64 ms 
TIME    92.47 ms 
End.
```

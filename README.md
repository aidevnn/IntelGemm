# IntelGemm
GEneral Matrix Multiplication with Intel Compiler and his powerfull Autoparalleziation and Autovectorization

The matrix multiplication is optimized and its not the common formula which is unusable for a large size matrix.
This optimization is used by the fortran BLAS library.

```

void MatMul0(int m, int n, int k, float* a, float* b, float* c) {
	for (int i = 0; i < m; ++i) {
		for (int k0 = 0; k0 < k; ++k0) {
			for (int j = 0; j < n; ++j)
				c[i * n + j] += a[i * k + k0] * b[k0 * n + j];
		}
	}
}

void MatMul1(int m, int n, int k, float* a, float* b, float* c) {
	cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0f, a, k, b, n, 0.0f, c, n);
}
```

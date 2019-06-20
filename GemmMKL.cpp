//============================================================================
// Name        : GemmMKL.cpp
// Author      : AIdevNN
// Version     :
// Copyright   : @2019
// Description : GEneral Matrix Multiplication using Intel MKL in C++, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <chrono>
#include <cmath>
#include <string>
#include "mkl.h"

using namespace std;
using namespace std::chrono;

high_resolution_clock::time_point now = high_resolution_clock::now();
#define TIME duration_cast<duration<double>>(high_resolution_clock::now() - now).count()

float getRand(bool is_rounded = true) {
	auto f = rand() / static_cast<float>(RAND_MAX) - 0.5;
	if (is_rounded)
		return round(f * 20.0);

	return f;
}

void fillRand(int m, int n, float Mat[], bool isrounded = true) {
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			Mat[i * n + j] = getRand(isrounded);
		}
	}
}

void displayMatrix(int m, int n, float* Mat, string name, bool isrounded = true) {
	printf("Matrix (%i %i) %s \n", m, n, name.c_str());
	string fmt = isrounded ? "%6.0f" : "%6.2f";
	for (int i = 0; i < m; ++i) {
		for (int j = 0; j < n; ++j) {
			printf(fmt.c_str(), Mat[i * n + j]);
		}
		printf("\n");
	}
	printf("\n");
}

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

void testGemm(int m, int n, int k) {

	float* A = (float *) malloc(m * k * sizeof(float));
	float* B = (float *) malloc(n * k * sizeof(float));
	float* C0 = (float *) malloc(m * n * sizeof(float));
	float* C1 = (float *) malloc(m * n * sizeof(float));
	float* C2 = (float *) malloc(m * n * sizeof(float));

	for (int i = 0; i < m * k; ++i)
		A[i] = getRand();

	for (int i = 0; i < k * n; ++i)
		B[i] = getRand();

	MatMul0(m, n, k, A, B, C0);
	MatMul1(m, n, k, A, B, C1);
	MatMul2(m, n, k, A, B, C2);
	displayMatrix(m, k, A, "A");
	displayMatrix(k, n, B, "B");
	displayMatrix(m, n, C0, "A x B meth0");
	displayMatrix(m, n, C1, "A x B meth1");
	displayMatrix(m, n, C2, "A x B meth2");

	free(A);
	free(B);
	free(C0);
	free(C1);
	free(C2);
}

void benchGemm(int m, int n, int k) {

	printf("Start Bench M=%i N=%i and K=%i \n", m, n, k);

	float* A = (float *) malloc(m * k * sizeof(float));
	float* B = (float *) malloc(n * k * sizeof(float));
	float* C0 = (float *) malloc(m * n * sizeof(float));
	float* C1 = (float *) malloc(m * n * sizeof(float));
	float* C2 = (float *) malloc(m * n * sizeof(float));

	for (int i = 0; i < m * k; ++i)
		A[i] = getRand(false);

	for (int i = 0; i < k * n; ++i)
		B[i] = getRand(false);

	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < m * n; ++j)
			C0[i] = C1[i] = C2[i] = 0.0f;

		MatMul0(m, n, k, A, B, C0);
		MatMul1(m, n, k, A, B, C1);
		MatMul2(m, n, k, A, B, C2);
	}

	cout << "Bench methode 0" << endl;
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < m * n; ++j)
			C0[i] = 0.0f;

		now = high_resolution_clock::now();
		MatMul0(m, n, k, A, B, C0);
		printf("TIME %8.2f ms \n", (round(TIME * 100000.0) * 0.01));
	}

	cout << "Bench methode 1" << endl;
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < m * n; ++j)
			C1[i] = 0.0f;

		now = high_resolution_clock::now();
		MatMul1(m, n, k, A, B, C1);
		printf("TIME %8.2f ms \n", (round(TIME * 100000.0) * 0.01));
	}

	cout << "Bench methode 2" << endl;
	for (int i = 0; i < 5; ++i) {
		for (int j = 0; j < m * n; ++j)
			C2[i] = 0.0f;

		now = high_resolution_clock::now();
		MatMul2(m, n, k, A, B, C2);
		printf("TIME %8.2f ms \n", (round(TIME * 100000.0) * 0.01));
	}

	cout << "End." << endl;
	cout << endl;

	free(A);
	free(B);
	free(C0);
	free(C1);
	free(C2);
}

int main() {
	cout << "HelloWorld MKL Gemm" << endl; // prints HelloWorld Test
	srand(time(NULL));

	testGemm(2, 4, 3);

//	benchGemm(96, 128, 64);
//	benchGemm(960, 1280, 640);
	benchGemm(1920, 2560, 1280);

	return 0;
}

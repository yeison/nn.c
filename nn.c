#include <stdio.h>
#include <math.h>
#include <omp.h>

typedef struct {
    int r; // rows
    int c; // columns
    double** data; // data in row major ordering
} Matrix2D;

// initialize an m*n matrix
Matrix2D* matrix_init(int m, int n) {
    Matrix2D* new_matrix = malloc(sizeof(Matrix2D));
    new_matrix->r = m;
    new_matrix->c = n;
    new_matrix->data = malloc(sizeof(double*) * m);
    for (int i = 0; i < m; i++) {
        new_matrix->data[i] = malloc(sizeof(double) * n);
    }

    return new_matrix;
}

void fill_rand(Matrix2D* A) {
    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < A->c; j++) {
            A->data[i][j] = rand();
        }
    }
}

void print(Matrix2D* A) {
    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < A->c; j++) {
            printf("%20f", A->data[i][j]);
        }
        printf("\n");
    }
}

void matmul(Matrix2D* A, Matrix2D* B) {
    if (A->c != B->r) {
        perror("Incompatible matrix dimensions");
        return;
    }

    Matrix2D* result = matrix_init(A->r, B->c);

    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < B->c; j++) {
            for (int k = 0; k < A->c; k++) {
                result->data[i][j] = A->data[i][k] * B->data[k][j];
            }
        }
    }
}

int main(int argc, char** argv) {
    Matrix2D* A = matrix_init(2, 2);

    fill_rand(A);

    print(A);
}
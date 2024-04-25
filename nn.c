#include <stdio.h>
#include <math.h>
#include <omp.h>

typedef double (*Operation)();

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

void fill(Matrix2D* A, Operation op) {
    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < A->c; j++) {
            A->data[i][j] = op();
        }
    }
}

double randf() {
    return (rand() % 10000) * 0.001;
}

double ones() {
    return 1;
}

double twos() {
    return 2;
}

void print(Matrix2D* A) {
    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < A->c; j++) {
            printf("%15f", A->data[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}

Matrix2D* init_with(int m, int n, Operation op) {
    Matrix2D* A = matrix_init(m, n);
    fill(A, op);
    print(A);
    return A;
}

Matrix2D* init_withr(int m, int n) {
    return init_with(m, n, randf);
}

// Transpose a matrix
Matrix2D* t(Matrix2D* A) {
    Matrix2D* transpose = matrix_init(A->c, A->r);

    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < A->c; j++) {
            transpose->data[j][i] = A->data[i][j];
        }
    }

    return transpose;   
}

// Multiply A by val in place
void mult_scalar(Matrix2D* A, double val) {
    for (int i = 0; i < A->r; i++){
        for (int j = 0; j < A->c; j++) {
            A->data[i][j] = A->data[i][j]  * val;
        }        
    }
}

// element-wise matrix multiplication
Matrix2D* element_mult(Matrix2D* A, Matrix2D* B) {
    if (A->r != B->r || A->c != B->c) {
        perror("Matrix dimensions must be the same");
        return NULL;
    }

    Matrix2D* result = matrix_init(A->r, A->c);
    for (int i = 0; i < A->r; i++){
        for (int j = 0; j < A->c; j++) {
            result->data[i][j] = A->data[i][j] * B->data[i][j];
        }        
    }
    
    return result;
}

Matrix2D* element_add(Matrix2D* A, Matrix2D* B) {
    Matrix2D* result = matrix_init(A->r, A->c);
    for (int i = 0; i < A->r; i++){
        for (int j = 0; j < A->c; j++) {
            result->data[i][j] = A->data[i][j] + B->data[0][j];
        }
    }

    return result;
}

// matrix dot product
Matrix2D* matmul(Matrix2D* A, Matrix2D* B) {
    if (A->c != B->r) {
        perror("Incompatible matrix dimensions");
        return NULL;
    }

    Matrix2D* result = matrix_init(A->r, B->c);
    for (int i = 0; i < A->r; i++) {
        for (int j = 0; j < B->c; j++) {
            for (int k = 0; k < A->c; k++) {
                result->data[i][j] += A->data[i][k] * B->data[k][j];
            }
        }
    }

    return result;
}

Matrix2D* forwardprop(Matrix2D* input, Matrix2D* weights, Matrix2D* bias) {
    Matrix2D* output = matmul(input, weights);
    output = element_add(output, bias);
    return output;
}

Matrix2D* backprop(Matrix2D* input, Matrix2D* weights, Matrix2D* bias, Matrix2D* output_error, double learning_rate) {
    Matrix2D* input_error = matmul(output_error, t(weights));
    Matrix2D* weights_error = matmul(t(input), output_error);

    mult_scalar(weights_error, learning_rate);
    mult_scalar(output_error, learning_rate);

    weights = weights_error;
    bias = output_error;
    
    return input_error;
}

int main(int argc, char** argv) {
    Matrix2D* A = init_with(1, 2, ones);
    Matrix2D* B = init_with(2, 2, ones);

    Matrix2D* C = matmul(A, B);

    print(C);
}
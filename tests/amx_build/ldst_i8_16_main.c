#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "ldst_i8_16_lib.h"

void print_2i8(int N, int M, int8_t *data) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++)
            printf("%d ", (int)data[M*i + j]);
        printf("\n");
    }
}
void print_4i8(int N, int M, int K, int R, int8_t *data) {
    printf("%d %d %d %d\n", N, M, K, R);
    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++) {
            printf("{ ");
            for(int k=0; k<K; k++) {
                printf("{ ");
                for(int r=0; r<R; r++)
                    printf("%d ", (int)data[M*K*R*i + K*R*j + R*k + r]);
                printf("}, ");
            }
            printf("}, ");
        }
        printf("\n");
    }
}
void print_2i32(int N, int M, int32_t *data) {
    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++)
            printf("%d ", (int)data[M*i + j]);
        printf("\n");
    }
}
bool check_eq_2i8(int N, int M, int8_t *lhs, int8_t *rhs) {
    bool flag = true;    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++)
            if(lhs[M*i + j] != rhs[M*i + j])
                flag = false;    }
    return flag;}
bool check_eq_4i8(int N, int M, int K, int R, int8_t *lhs, int8_t *rhs) {
    bool flag = true;    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++)
            for(int k=0; k<K; k++)
                for(int r=0; r<R; r++)
                    if(lhs[M*K*R*i + K*R*j + R*k + r] != rhs[M*K*R*i + K*R*j + R*k + r])
                        flag = false;    }
    return flag;}
bool check_eq_2i32(int N, int M, int32_t *lhs, int32_t *rhs) {
    bool flag = true;    for(int i=0; i<N; i++) {
        for(int j=0; j<M; j++)
            if(lhs[M*i + j] != rhs[M*i + j])
                flag = false;    }
    return flag;}

int8_t x[16*16];

int8_t y[16*16];


int main() {
    ldst_i8_16_lib_Context *ctxt;
    for(int i=0; i<16; i++) {
        for(int j=0; j<16; j++) {
            x[(16)*i + j] = i+j;
    }}
    
    for(int i=0; i<16; i++) {
        for(int j=0; j<16; j++) {
            y[(16)*i + j] = 0;
    }}
    
    ldst_i8_16(ctxt, x, y);
    
    if(check_eq_2i8(16,16, x, y)) {
        printf("Correct\n");
    } else {
        printf("Results Don't Match\n");
        printf("Correct Result (x):\n");
        print_2i8(16,16, x);
        printf("Computed Roundtrip (y):\n");
        print_2i8(16,16, y);
        exit(1);
    }
    

    printf("\nDone\n");
    
    exit(0);
}

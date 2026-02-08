#ifndef TENSOR_H
#define TENSOR_H

#include "base.h"

// ============================================
// Tensor Types
// ============================================

typedef size_t tensor_size_t;

typedef struct tensor_t {
    tensor_size_t *shape;
    tensor_size_t dimensions;
    tensor_size_t *strides;
    float32_t *data;
} tensor_t;

// ============================================
// Core Operations
// ============================================

tensor_t *tensor_new(arena_t *arena, tensor_size_t *shape, tensor_size_t dimensions) {
    assert(arena != NULL);
    assert(shape != NULL);
    assert(dimensions != 0);

    tensor_t *array = (tensor_t *)arena_alloc(arena, sizeof(tensor_t));
    array->dimensions = dimensions;
    array->shape = (tensor_size_t *)arena_alloc(arena, sizeof(tensor_size_t) * dimensions);
    array->strides = (tensor_size_t *)arena_alloc(arena, sizeof(tensor_size_t) * dimensions);

    for (tensor_size_t i = 0; i < dimensions; i++) {
        array->shape[i] = shape[i];
    }

    tensor_size_t total_size = 1;
    for (tensor_size_t i = dimensions; i > 0; i--) {
        array->strides[i - 1] = total_size;
        total_size *= shape[i - 1];
    }

    array->data = (float32_t *)arena_alloc(arena, sizeof(float32_t) * total_size);
    return array;
}

tensor_size_t compute_offset(tensor_t *array, tensor_size_t *indices) {
    size_t offset = 0;
    for (size_t i = 0; i < array->dimensions; i++) {
        offset += indices[i] * array->strides[i];
    }
    return offset;
}

float32_t tensor_get(tensor_t *array, tensor_size_t *indices) {
    return array->data[compute_offset(array, indices)];
}

void tensor_set(tensor_t *array, tensor_size_t *indices, float32_t value) {
    array->data[compute_offset(array, indices)] = value;
}

tensor_size_t tensor_total_size(tensor_t *array) {
    tensor_size_t total = 1;
    for (size_t i = 0; i < array->dimensions; i++) {
        total *= array->shape[i];
    }
    return total;
}

// ============================================
// Fill Operations
// ============================================

void tensor_fill(tensor_t *array, float32_t value) {
    tensor_size_t total = tensor_total_size(array);
    for (tensor_size_t i = 0; i < total; i++) {
        array->data[i] = value;
    }
}

void tensor_zeros(tensor_t *array) { tensor_fill(array, 0.0f); }
void tensor_ones(tensor_t *array) { tensor_fill(array, 1.0f); }

void tensor_randn(tensor_t *t) {
    tensor_size_t total = tensor_total_size(t);
    for (tensor_size_t i = 0; i < total; i++) {
        t->data[i] = random_normal();
    }
}

void tensor_copy(tensor_t *src, tensor_t *dst) {
    tensor_size_t total = tensor_total_size(src);
    for (tensor_size_t i = 0; i < total; i++) {
        dst->data[i] = src->data[i];
    }
}

// ============================================
// Element-wise Operations
// ============================================

void tensor_add(tensor_t *a, tensor_t *b, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] + b->data[i];
    }
}

void tensor_sub(tensor_t *a, tensor_t *b, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] - b->data[i];
    }
}

void tensor_mul(tensor_t *a, tensor_t *b, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] * b->data[i];
    }
}

void tensor_div(tensor_t *a, tensor_t *b, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] / b->data[i];
    }
}

void tensor_scale(tensor_t *a, float32_t scalar, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] * scalar;
    }
}

void tensor_add_scalar(tensor_t *a, float32_t scalar, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] + scalar;
    }
}

void tensor_clip(tensor_t *t, float32_t min_val, float32_t max_val) {
    tensor_size_t total = tensor_total_size(t);
    for (tensor_size_t i = 0; i < total; i++) {
        if (t->data[i] < min_val) t->data[i] = min_val;
        if (t->data[i] > max_val) t->data[i] = max_val;
    }
}

// ============================================
// Reductions
// ============================================

float32_t tensor_sum(tensor_t *array) {
    float32_t sum = 0.0f;
    tensor_size_t total = tensor_total_size(array);
    for (tensor_size_t i = 0; i < total; i++) {
        sum += array->data[i];
    }
    return sum;
}

float32_t tensor_mean(tensor_t *array) {
    return tensor_sum(array) / (float32_t)tensor_total_size(array);
}

float32_t tensor_max(tensor_t *array) {
    float32_t max = array->data[0];
    tensor_size_t total = tensor_total_size(array);
    for (tensor_size_t i = 1; i < total; i++) {
        if (array->data[i] > max) max = array->data[i];
    }
    return max;
}

float32_t tensor_min(tensor_t *array) {
    float32_t min = array->data[0];
    tensor_size_t total = tensor_total_size(array);
    for (tensor_size_t i = 1; i < total; i++) {
        if (array->data[i] < min) min = array->data[i];
    }
    return min;
}

size_t tensor_argmax(tensor_t *t) {
    size_t idx = 0;
    float32_t max_val = t->data[0];
    tensor_size_t total = tensor_total_size(t);
    for (tensor_size_t i = 1; i < total; i++) {
        if (t->data[i] > max_val) {
            max_val = t->data[i];
            idx = i;
        }
    }
    return idx;
}

// ============================================
// Activations
// ============================================

void tensor_relu(tensor_t *a, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = a->data[i] > 0.0f ? a->data[i] : 0.0f;
    }
}

void tensor_sigmoid(tensor_t *a, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = 1.0f / (1.0f + expf(-a->data[i]));
    }
}

void tensor_tanh(tensor_t *a, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = tanhf(a->data[i]);
    }
}

void tensor_softmax(tensor_t *a, tensor_t *out) {
    tensor_size_t total = tensor_total_size(a);
    float32_t max_val = tensor_max(a);
    float32_t sum = 0.0f;
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] = expf(a->data[i] - max_val);
        sum += out->data[i];
    }
    for (tensor_size_t i = 0; i < total; i++) {
        out->data[i] /= sum;
    }
}

// ============================================
// Activation Gradients
// ============================================

void tensor_relu_grad(tensor_t *a, tensor_t *grad_out, tensor_t *grad_in) {
    tensor_size_t total = tensor_total_size(a);
    for (tensor_size_t i = 0; i < total; i++) {
        grad_in->data[i] = a->data[i] > 0.0f ? grad_out->data[i] : 0.0f;
    }
}

void tensor_sigmoid_grad(tensor_t *sigmoid_out, tensor_t *grad_out, tensor_t *grad_in) {
    tensor_size_t total = tensor_total_size(sigmoid_out);
    for (tensor_size_t i = 0; i < total; i++) {
        float32_t s = sigmoid_out->data[i];
        grad_in->data[i] = grad_out->data[i] * s * (1.0f - s);
    }
}

void tensor_tanh_grad(tensor_t *tanh_out, tensor_t *grad_out, tensor_t *grad_in) {
    tensor_size_t total = tensor_total_size(tanh_out);
    for (tensor_size_t i = 0; i < total; i++) {
        float32_t t = tanh_out->data[i];
        grad_in->data[i] = grad_out->data[i] * (1.0f - t * t);
    }
}

void tensor_relu_backward_inplace(tensor_t *x, float32_t *dx, size_t size) {
    for (size_t i = 0; i < size; i++) {
        dx[i] = (x->data[i] > 0.0f) ? dx[i] : 0.0f;
    }
}

// ============================================
// Loss Functions
// ============================================

float32_t tensor_mse_loss(tensor_t *pred, tensor_t *target) {
    tensor_size_t total = tensor_total_size(pred);
    float32_t loss = 0.0f;
    for (tensor_size_t i = 0; i < total; i++) {
        float32_t diff = pred->data[i] - target->data[i];
        loss += diff * diff;
    }
    return loss / (float32_t)total;
}

void tensor_mse_loss_grad(tensor_t *pred, tensor_t *target, tensor_t *grad) {
    tensor_size_t total = tensor_total_size(pred);
    float32_t scale = 2.0f / (float32_t)total;
    for (tensor_size_t i = 0; i < total; i++) {
        grad->data[i] = scale * (pred->data[i] - target->data[i]);
    }
}

float32_t tensor_cross_entropy_loss(tensor_t *pred, tensor_t *target) {
    tensor_size_t total = tensor_total_size(pred);
    float32_t loss = 0.0f;
    for (tensor_size_t i = 0; i < total; i++) {
        if (target->data[i] > 0.0f) {
            loss -= target->data[i] * logf(pred->data[i] + 1e-7f);
        }
    }
    return loss;
}

void tensor_softmax_cross_entropy_grad(tensor_t *softmax_out, tensor_t *target, tensor_t *grad) {
    tensor_size_t total = tensor_total_size(softmax_out);
    for (tensor_size_t i = 0; i < total; i++) {
        grad->data[i] = softmax_out->data[i] - target->data[i];
    }
}

// ============================================
// Matrix Operations
// ============================================

void tensor_matmul(tensor_t *a, tensor_t *b, tensor_t *out,
                   tensor_size_t M, tensor_size_t K, tensor_size_t N) {
    for (tensor_size_t i = 0; i < M; i++) {
        for (tensor_size_t j = 0; j < N; j++) {
            float32_t sum = 0.0f;
            for (tensor_size_t k = 0; k < K; k++) {
                sum += a->data[i * K + k] * b->data[k * N + j];
            }
            out->data[i * N + j] = sum;
        }
    }
}

void tensor_matmul_grad_a(tensor_t *grad_out, tensor_t *b, tensor_t *grad_a,
                          tensor_size_t M, tensor_size_t K, tensor_size_t N) {
    for (tensor_size_t i = 0; i < M; i++) {
        for (tensor_size_t k = 0; k < K; k++) {
            float32_t sum = 0.0f;
            for (tensor_size_t j = 0; j < N; j++) {
                sum += grad_out->data[i * N + j] * b->data[k * N + j];
            }
            grad_a->data[i * K + k] = sum;
        }
    }
}

void tensor_matmul_grad_b(tensor_t *a, tensor_t *grad_out, tensor_t *grad_b,
                          tensor_size_t M, tensor_size_t K, tensor_size_t N) {
    for (tensor_size_t k = 0; k < K; k++) {
        for (tensor_size_t j = 0; j < N; j++) {
            float32_t sum = 0.0f;
            for (tensor_size_t i = 0; i < M; i++) {
                sum += a->data[i * K + k] * grad_out->data[i * N + j];
            }
            grad_b->data[k * N + j] = sum;
        }
    }
}

// ============================================
// Linear Layer
// ============================================

void tensor_linear(float32_t *x, tensor_t *W, tensor_t *b, tensor_t *out,
                   size_t in_features, size_t out_features) {
    for (size_t j = 0; j < out_features; j++) {
        float32_t sum = b->data[j];
        for (size_t i = 0; i < in_features; i++) {
            sum += x[i] * W->data[i * out_features + j];
        }
        out->data[j] = sum;
    }
}

void tensor_linear_backward(float32_t *x, tensor_t *W, tensor_t *dout,
                            tensor_t *dW, tensor_t *db, float32_t *dx,
                            size_t in_features, size_t out_features) {
    for (size_t i = 0; i < in_features; i++) {
        for (size_t j = 0; j < out_features; j++) {
            dW->data[i * out_features + j] = x[i] * dout->data[j];
        }
    }
    for (size_t j = 0; j < out_features; j++) {
        db->data[j] = dout->data[j];
    }
    if (dx != NULL) {
        for (size_t i = 0; i < in_features; i++) {
            float32_t sum = 0.0f;
            for (size_t j = 0; j < out_features; j++) {
                sum += dout->data[j] * W->data[i * out_features + j];
            }
            dx[i] = sum;
        }
    }
}

// ============================================
// Optimizers
// ============================================

void tensor_accumulate_grad(tensor_t *grad, tensor_t *delta) {
    tensor_size_t total = tensor_total_size(grad);
    for (tensor_size_t i = 0; i < total; i++) {
        grad->data[i] += delta->data[i];
    }
}

void tensor_sgd_update(tensor_t *param, tensor_t *grad, float32_t lr) {
    tensor_size_t total = tensor_total_size(param);
    for (tensor_size_t i = 0; i < total; i++) {
        param->data[i] -= lr * grad->data[i];
    }
}

// ============================================
// Initialization
// ============================================

void tensor_init_xavier(tensor_t *t, size_t fan_in, size_t fan_out) {
    float32_t scale = sqrtf(2.0f / (float32_t)(fan_in + fan_out));
    tensor_size_t total = tensor_total_size(t);
    for (tensor_size_t i = 0; i < total; i++) {
        t->data[i] = random_normal() * scale;
    }
}

void tensor_init_he(tensor_t *t, size_t fan_in) {
    float32_t scale = sqrtf(2.0f / (float32_t)fan_in);
    tensor_size_t total = tensor_total_size(t);
    for (tensor_size_t i = 0; i < total; i++) {
        t->data[i] = random_normal() * scale;
    }
}

#endif // TENSOR_H

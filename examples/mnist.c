#include "../tensor.h"
#include <time.h>

typedef struct mnist_t
{
    tensor_t *images;
    tensor_t *labels; // [N, 10] one-hot
    size_t num_samples;
} mnist_t;

mnist_t *mnist_load(arena_t *arena, const char *images_path, const char *labels_path)
{
    mnist_t *mnist = (mnist_t *)arena_alloc(arena, sizeof(mnist_t));

    // load image
    FILE *img_file = fopen(images_path, "rb");
    if (!img_file)
    {
        printf("Error: Could not open %s\n", images_path);
        return NULL;
    }

    uint32_t magic, num_images, rows, cols;
    fread(&magic, 4, 1, img_file);
    fread(&num_images, 4, 1, img_file);
    fread(&rows, 4, 1, img_file);
    fread(&cols, 4, 1, img_file);

    magic = swap_endian(magic);
    num_images = swap_endian(num_images);
    rows = swap_endian(rows);
    cols = swap_endian(cols);

    printf("Images: %u, %ux%u\n", num_images, rows, cols);

    mnist->num_samples = num_images;
    tensor_size_t img_shape[] = {num_images, 784};
    mnist->images = tensor_new(arena, img_shape, 2);

    uint8_t *pixel_buffer = (uint8_t *)arena_alloc(arena, 784);
    for (size_t i = 0; i < num_images; i++)
    {
        fread(pixel_buffer, 1, 784, img_file);
        for (size_t j = 0; j < 784; j++)
        {
            mnist->images->data[i * 784 + j] = pixel_buffer[j] / 255.0f;
        }
    }
    fclose(img_file);

    // load labels
    FILE *lbl_file = fopen(labels_path, "rb");
    if (!lbl_file)
    {
        printf("Error: Could not open %s\n", labels_path);
        return NULL;
    }

    uint32_t lbl_magic, num_labels;
    fread(&lbl_magic, 4, 1, lbl_file);
    fread(&num_labels, 4, 1, lbl_file);
    lbl_magic = swap_endian(lbl_magic);
    num_labels = swap_endian(num_labels);

    tensor_size_t lbl_shape[] = {num_labels, 10};
    mnist->labels = tensor_new(arena, lbl_shape, 2);
    tensor_zeros(mnist->labels);

    for (size_t i = 0; i < num_labels; i++)
    {
        uint8_t label;
        fread(&label, 1, 1, lbl_file);
        mnist->labels->data[i * 10 + label] = 1.0f;
    }
    fclose(lbl_file);

    return mnist;
}

// ============================================
// MLP: 784 -> 128 -> 10
// ============================================

typedef struct mlp_t
{
    // Weights and biases
    tensor_t *w1; // [784, 128]
    tensor_t *b1; // [128]
    tensor_t *w2; // [128, 10]
    tensor_t *b2; // [10]

    // Gradients
    tensor_t *dw1;
    tensor_t *db1;
    tensor_t *dw2;
    tensor_t *db2;

    // Activations (for backprop)
    tensor_t *z1; // [128] pre-activation
    tensor_t *a1; // [128] post-ReLU
    tensor_t *z2; // [10] pre-softmax
    tensor_t *a2; // [10] output (softmax)

    // Intermediate gradients
    tensor_t *dz1;
    tensor_t *dz2;
} mlp_t;

mlp_t *mlp_new(arena_t *arena)
{
    mlp_t *mlp = (mlp_t *)arena_alloc(arena, sizeof(mlp_t));

    // Weights
    tensor_size_t w1_shape[] = {784, 128};
    tensor_size_t b1_shape[] = {128};
    tensor_size_t w2_shape[] = {128, 10};
    tensor_size_t b2_shape[] = {10};

    mlp->w1 = tensor_new(arena, w1_shape, 2);
    mlp->b1 = tensor_new(arena, b1_shape, 1);
    mlp->w2 = tensor_new(arena, w2_shape, 2);
    mlp->b2 = tensor_new(arena, b2_shape, 1);

    // Initialize weights
    tensor_init_xavier(mlp->w1, 784, 128);
    tensor_init_xavier(mlp->w2, 128, 10);
    tensor_zeros(mlp->b1);
    tensor_zeros(mlp->b2);

    // Gradients
    mlp->dw1 = tensor_new(arena, w1_shape, 2);
    mlp->db1 = tensor_new(arena, b1_shape, 1);
    mlp->dw2 = tensor_new(arena, w2_shape, 2);
    mlp->db2 = tensor_new(arena, b2_shape, 1);

    // Activations
    mlp->z1 = tensor_new(arena, b1_shape, 1);
    mlp->a1 = tensor_new(arena, b1_shape, 1);
    mlp->z2 = tensor_new(arena, b2_shape, 1);
    mlp->a2 = tensor_new(arena, b2_shape, 1);

    // Intermediate gradients
    mlp->dz1 = tensor_new(arena, b1_shape, 1);
    mlp->dz2 = tensor_new(arena, b2_shape, 1);

    return mlp;
}

void mlp_forward(mlp_t *mlp, float32_t *input)
{
    // z1 = input @ W1 + b1
    tensor_linear(input, mlp->w1, mlp->b1, mlp->z1, 784, 128);

    // a1 = ReLU(z1)
    tensor_relu(mlp->z1, mlp->a1);

    // z2 = a1 @ W2 + b2
    tensor_linear(mlp->a1->data, mlp->w2, mlp->b2, mlp->z2, 128, 10);

    // a2 = softmax(z2)
    tensor_softmax(mlp->z2, mlp->a2);
}

void mlp_backward(mlp_t *mlp, float32_t *input, float32_t *target)
{
    // dz2 = a2 - target (softmax + cross-entropy gradient)
    for (size_t i = 0; i < 10; i++)
    {
        mlp->dz2->data[i] = mlp->a2->data[i] - target[i];
    }

    // Layer 2 backward: dW2, db2, and da1
    tensor_linear_backward(mlp->a1->data, mlp->w2, mlp->dz2,
                           mlp->dw2, mlp->db2, mlp->dz1->data,
                           128, 10);

    // dz1 = da1 * ReLU'(z1)
    tensor_relu_backward_inplace(mlp->z1, mlp->dz1->data, 128);

    // Layer 1 backward: dW1, db1 (no need for dx)
    tensor_linear_backward(input, mlp->w1, mlp->dz1,
                           mlp->dw1, mlp->db1, NULL,
                           784, 128);
}

void mlp_update(mlp_t *mlp, float32_t lr)
{
    tensor_sgd_update(mlp->w1, mlp->dw1, lr);
    tensor_sgd_update(mlp->b1, mlp->db1, lr);
    tensor_sgd_update(mlp->w2, mlp->dw2, lr);
    tensor_sgd_update(mlp->b2, mlp->db2, lr);
}

size_t mlp_predict(mlp_t *mlp)
{
    size_t pred = 0;
    float32_t max_val = mlp->a2->data[0];
    for (size_t i = 1; i < 10; i++)
    {
        if (mlp->a2->data[i] > max_val)
        {
            max_val = mlp->a2->data[i];
            pred = i;
        }
    }
    return pred;
}

size_t label_to_class(float32_t *label)
{
    for (size_t i = 0; i < 10; i++)
    {
        if (label[i] > 0.5f)
            return i;
    }
    return 0;
}

// ============================================
// Main
// ============================================

int main(int argc, char **argv)
{
    srand((unsigned int)time(NULL));

    arena_t arena;
    arena_init(&arena, MiB(512));

    printf("Loading MNIST...\n");
    mnist_t *train = mnist_load(&arena,
                                "data/train-images.idx3-ubyte",
                                "data/train-labels.idx1-ubyte");

    if (!train)
    {
        printf("Error loading data. Make sure MNIST files are in data/\n");
        return 1;
    }

    printf("Creating model...\n");
    mlp_t *mlp = mlp_new(&arena);

    // Hyperparameters
    float32_t lr = 0.01f;
    size_t epochs = 5;
    size_t batch_report = 10000;

    printf("Training...\n");
    for (size_t epoch = 0; epoch < epochs; epoch++)
    {
        float32_t total_loss = 0.0f;
        size_t correct = 0;

        for (size_t i = 0; i < train->num_samples; i++)
        {
            float32_t *input = &train->images->data[i * 784];
            float32_t *target = &train->labels->data[i * 10];

            // Forward
            mlp_forward(mlp, input);

            // Loss
            for (size_t j = 0; j < 10; j++)
            {
                if (target[j] > 0.5f)
                {
                    total_loss -= logf(mlp->a2->data[j] + 1e-7f);
                }
            }

            // Accuracy
            if (mlp_predict(mlp) == label_to_class(target))
            {
                correct++;
            }

            // Backward
            mlp_backward(mlp, input, target);

            // Update
            mlp_update(mlp, lr);

            if ((i + 1) % batch_report == 0)
            {
                printf("  Epoch %zu, Sample %zu/%zu, Loss: %.4f, Acc: %.2f%%\n",
                       epoch + 1, i + 1, train->num_samples,
                       total_loss / (i + 1),
                       100.0f * correct / (i + 1));
            }
        }

        printf("Epoch %zu completed - Loss: %.4f, Accuracy: %.2f%%\n",
               epoch + 1,
               total_loss / train->num_samples,
               100.0f * correct / train->num_samples);
    }

    printf("Training completed!\n");

    return 0;
}

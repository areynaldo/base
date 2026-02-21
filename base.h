#ifndef BASE_H
#define BASE_H

#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <memory.h>
#include <string.h>
#include <math.h>

// ============================================
// Types
// ============================================

typedef float float32_t;
typedef double float64_t;

// ============================================
// Macros
// ============================================

#define XCONCAT(a, b) a##b
#define CONCAT(a, b) XCONCAT(a, b)
#define STRINGIFY(a) TO_STRING(a)
#define TO_STRING(a) #a
#define ARRAY_LENGTH(arr) (sizeof((arr)) / sizeof((arr)[0]))

#define KiB(x) ((x) * 1024LL)
#define MiB(x) (KiB((x)) * 1024LL)
#define GiB(x) (MiB((x)) * 1024LL)

#ifndef MEMORY_DEFAULT_ALIGNMENT
#define MEMORY_DEFAULT_ALIGNMENT (2 * sizeof(void *))
#endif

// ============================================
// Errors
// ============================================

#define ERRORS_X \
    ERROR_X(NONE, "No error.") \
    ERROR_X(ARENA_INIT, "Arena init error.") \
    ERROR_X(ARENA_ALLOC, "Arena alloc error.") \
    ERROR_X(FILE_OPEN, "File open error.") \
    ERROR_X(COUNT, "Count.")

typedef enum error_t {
#define ERROR_X(r, m) CONCAT(ERROR_, r),
    ERRORS_X
#undef ERROR_X
} error_t;

static const char *error_messages[] = {
#define ERROR_X(r, m) m,
    ERRORS_X
#undef ERROR_X
};

// ============================================
// Memory
// ============================================

bool is_power_of_two(uintptr_t x) {
    return (x & (x - 1)) == 0;
}

void *memory_align_forward(void *ptr, size_t alignment) {
    assert(ptr);
    assert(is_power_of_two(alignment));
    uintptr_t result = (uintptr_t)ptr;
    uintptr_t modulo = result & (alignment - 1);
    if (modulo != 0) {
        result += alignment - modulo;
    }
    return (void *)result;
}

typedef struct arena_t {
    uint8_t *data;
    size_t current_offset;
    size_t capacity;
} arena_t;

error_t arena_init(arena_t *arena, size_t size) {
    assert(arena != NULL);
    assert(size != 0);
    arena->data = (uint8_t *)malloc(size);
    if (!arena->data) return ERROR_ARENA_INIT;
    arena->capacity = size;
    arena->current_offset = 0;
    return ERROR_NONE;
}

void *arena_alloc_align(arena_t *arena, size_t size, size_t alignment) {
    assert(arena != NULL);
    assert(size != 0);
    assert(is_power_of_two(alignment));

    uintptr_t curr_ptr = (uintptr_t)arena->data + (uintptr_t)arena->current_offset;
    uintptr_t offset = (uintptr_t)memory_align_forward((void *)curr_ptr, alignment);
    offset -= (uintptr_t)arena->data;

    if (offset + size <= arena->capacity) {
        void *result = &arena->data[offset];
        arena->current_offset = offset + size;
        memset(result, 0, size);
        return result;
    }
    return NULL;
}

void *arena_alloc(arena_t *arena, size_t size) {
    return arena_alloc_align(arena, size, MEMORY_DEFAULT_ALIGNMENT);
}

void arena_reset(arena_t *arena) {
    arena->current_offset = 0;
}

// ============================================
// Strings
// ============================================

typedef struct string_t {
    char *str;
    size_t length;
} string_t;

error_t string_from_file(arena_t *arena, string_t *string, const char *path) {
    FILE *file = fopen(path, "rb");
    if (!file) return ERROR_FILE_OPEN;

    fseek(file, 0, SEEK_END);
    size_t size = ftell(file);
    fseek(file, 0, SEEK_SET);

    string->str = (char *)arena_alloc(arena, size + 1);
    if (!string->str) {
        fclose(file);
        return ERROR_ARENA_ALLOC;
    }

    fread(string->str, 1, size, file);
    string->str[size] = '\0';
    string->length = size;
    fclose(file);
    return ERROR_NONE;
}

// ============================================
// Utilities
// ============================================

uint32_t swap_endian(uint32_t val) {
    return ((val >> 24) & 0xff) |
           ((val >> 8) & 0xff00) |
           ((val << 8) & 0xff0000) |
           ((val << 24) & 0xff000000);
}

float32_t random_uniform() {
    return (float32_t)rand() / RAND_MAX;
}

float32_t random_normal() {
    float32_t u1 = random_uniform();
    float32_t u2 = random_uniform();
    return sqrtf(-2.0f * logf(u1 + 1e-7f)) * cosf(2.0f * 3.14159265f * u2);
}

#endif // BASE_H
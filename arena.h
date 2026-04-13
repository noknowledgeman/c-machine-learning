#ifndef ARENA_H
#define ARENA_H

#include <stddef.h> 
#include <stdlib.h>
#include <string.h>

// 128MB initial arena size
#define ARENA_INITIAL_SIZE (1024*1024*128)

typedef struct {
    size_t cap;
    size_t len;
    char *data;
} Arena;

// 0 initialized
static Arena arenaCreate() {
    char *data = (char *)calloc(ARENA_INITIAL_SIZE, 1);
    return (Arena){
        .cap = ARENA_INITIAL_SIZE,
        .len = 0,
        .data = data,
    };
}

// zero-initialized for now for convenience
static void *arenaAlloc(Arena *arena, size_t size) {
    if (arena->len + size > arena->cap) return NULL;
    size_t loc = arena->len;
    arena->len += size;
    memset(arena->data + loc, 0, size);
    return arena->data + loc;
}

static void arenaReset(Arena *arena) {
    arena->len = 0;
}

static void arenaDestroy(Arena *arena) {
    free(arena->data);
}

#endif
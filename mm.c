/*
 * Simple, 64-bit allocator based on implicit free lists,
 * first fit placement, and boundary tag coalescing, as described
 * in the CS:APP2e text. Blocks must be aligned to 16 byte
 * boundaries. Minimum block size is 16 bytes.
 *
 * This version is loosely based on
 * http://csapp.cs.cmu.edu/3e/ics3/code/vm/malloc/mm.c
 * but unlike the book's version, it does not use C preprocessor
 * macros or explicit bit operations.
 *
 * It follows the book in counting in units of 4-byte words,
 * but note that this is a choice (my actual solution chooses
 * to count everything in bytes instead.)
 *
 * You should use this code as a starting point for your
 * implementation.
 *
 * First adapted for CS3214 Summer 2020 by gback
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

#include "mm.h"
#include "memlib.h"
#include "config.h"
#include "list.h"

/*
Notes:
- Base tests can pass with explicit list through init/malloc/free/coalesce/place
- Implement realloc after tests pass with explicit list

Current version: segregated list base tests passing by modifying init/coalesce/place/find_fit, left malloc/free alone

Current Score:
Results for mm malloc:
trace                  name valid util     ops      secs  Kops
 0           amptjp-bal.rep  yes   97%   56940  0.002552 22315
 1             cccp-bal.rep  yes   93%   58480  0.002879 20310
 2          cp-decl-bal.rep  yes   98%   66480  0.003185 20876
 3             expr-bal.rep  yes   99%   53800  0.002443 22019
 4       coalescing-bal.rep  yes   99%  144000  0.004110 35040
 5           random-bal.rep  yes   92%   48000  0.003553 13511
 6          random2-bal.rep  yes   91%   48000  0.003647 13163
 7           binary-bal.rep  yes   54%  120000  0.136198   881
 8          binary2-bal.rep  yes   47%  240000  0.804384   298
 9          realloc-bal.rep  yes   29%  144010  0.682794   211
10         realloc2-bal.rep  yes   27%  144010  0.024234  5943
Total                              75% 1123720  1.669978   673

Perf index = 45 (util) + 1 (thru) = 46/100
*/

struct boundary_tag
{
    int inuse : 1; // inuse bit
    int size : 31; // size of block, in words
                   // block size
};

/* FENCE is used for heap prologue/epilogue. */
const struct boundary_tag FENCE = {
    .inuse = -1,
    .size = 0};

/* A C struct describing the beginning of each block.
 * For implicit lists, used and free blocks have the same
 * structure, so one struct will suffice for this example.
 *
 * If each block is aligned at 12 mod 16, each payload will
 * be aligned at 0 mod 16.
 */
struct block
{
    struct boundary_tag header; /* offset 0, at address 12 mod 16 */
    char payload[0];            /* offset 4, at address 0 mod 16 */
    struct list_elem elem;
};

/* Basic constants and macros */
#define WSIZE sizeof(struct boundary_tag) /* Word and header/footer size (bytes) */
// Changed to 8 because the block struct has list elem in it
#define MIN_BLOCK_SIZE_WORDS 8 /* Minimum block size in words */
// Changed to 4 to increase util%
#define CHUNKSIZE (1 << 6) /* Extend heap by this amount (words) */

static inline size_t max(size_t x, size_t y)
{
    return x > y ? x : y;
}

static size_t align(size_t size)
{
    return (size + ALIGNMENT - 1) & ~(ALIGNMENT - 1);
}

static bool is_aligned(size_t size) __attribute__((__unused__));
static bool is_aligned(size_t size)
{
    return size % ALIGNMENT == 0;
}

/* Global variables */
static struct block *heap_listp = 0; /* Pointer to first block */
// static struct list free_list;
#define NUM_FREE_LISTS 16
static struct list free_lists[NUM_FREE_LISTS];

/* Helper functions*/
static int get_list_size(size_t size);

/* Function prototypes for internal helper routines */
static struct block *extend_heap(size_t words);
static void place(struct block *bp, size_t asize);
static struct block *find_fit(size_t asize);
static struct block *coalesce(struct block *bp);

/* Given a block, obtain previous's block footer.
   Works for left-most block also. */
static struct boundary_tag *prev_blk_footer(struct block *blk)
{
    return &blk->header - 1;
}

/* Return if block is free */
static bool blk_free(struct block *blk)
{
    return !blk->header.inuse;
}

/* Return size of block is free */
static size_t blk_size(struct block *blk)
{
    return blk->header.size;
}

/* Given a block, obtain pointer to previous block.
   Not meaningful for left-most block. */
static struct block *prev_blk(struct block *blk)
{
    struct boundary_tag *prevfooter = prev_blk_footer(blk);
    assert(prevfooter->size != 0);
    return (struct block *)((void *)blk - WSIZE * prevfooter->size);
}

/* Given a block, obtain pointer to next block.
   Not meaningful for right-most block. */
static struct block *next_blk(struct block *blk)
{
    assert(blk_size(blk) != 0);
    return (struct block *)((void *)blk + WSIZE * blk->header.size);
}

/* Given a block, obtain its footer boundary tag */
static struct boundary_tag *get_footer(struct block *blk)
{
    return ((void *)blk + WSIZE * blk->header.size) - sizeof(struct boundary_tag);
}

/* Set a block's size and inuse bit in header and footer */
static void set_header_and_footer(struct block *blk, int size, int inuse)
{
    blk->header.inuse = inuse;
    blk->header.size = size;
    *get_footer(blk) = blk->header; /* Copy header to footer */
}

/* Mark a block as used and set its size. */
static void mark_block_used(struct block *blk, int size)
{
    set_header_and_footer(blk, size, 1);
}

/* Mark a block as free and set its size. */
static void mark_block_free(struct block *blk, int size)
{
    set_header_and_footer(blk, size, 0);
}

/*
 * mm_init - Initialize the memory manager
 */
int mm_init(void)
{
    assert(offsetof(struct block, payload) == 4);
    assert(sizeof(struct boundary_tag) == 4);

    //  list_init(&free_list);
    for (int i = 0; i < NUM_FREE_LISTS; i++)
    {
        list_init(&free_lists[i]);
    }

    /* Create the initial empty heap */
    struct boundary_tag *initial = mem_sbrk(4 * sizeof(struct boundary_tag));
    if (initial == NULL)
        return -1;

    /* We use a slightly different strategy than suggested in the book.
     * Rather than placing a min-sized prologue block at the beginning
     * of the heap, we simply place two fences.
     * The consequence is that coalesce() must call prev_blk_footer()
     * and not prev_blk() because prev_blk() cannot be called on the
     * left-most block.
     */
    initial[2] = FENCE; /* Prologue footer */
    heap_listp = (struct block *)&initial[3];
    initial[3] = FENCE; /* Epilogue header */

    /* Extend the empty heap with a free block of CHUNKSIZE bytes */
    if (extend_heap(CHUNKSIZE) == NULL)
        return -1;
    return 0;
}

/*
 * mm_malloc - Allocate a block with at least size bytes of payload
 */
void *mm_malloc(size_t size)
{
    struct block *bp;

    /* Ignore spurious requests */
    if (size == 0)
        return NULL;

    /* Adjust block size to include overhead and alignment reqs. */
    size_t bsize = align(size + 2 * sizeof(struct boundary_tag)); /* account for tags */
    if (bsize < size)
        return NULL; /* integer overflow */

    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, bsize / WSIZE); /* respect minimum size */

    /* Search the free list for a fit */
    if ((bp = find_fit(awords)) != NULL)
    {
        place(bp, awords);
        // Assertion 1 -- Check to ensure the block payload is correctly alligned
        assert(((uintptr_t)bp->payload & (ALIGNMENT - 1)) == 0);
        return bp->payload;
    }

    /* No fit found. Get more memory and place the block */
    size_t extendwords = max(awords, CHUNKSIZE); /* Amount to extend heap if no fit */
    if ((bp = extend_heap(extendwords)) == NULL)
        return NULL;

    place(bp, awords);
    return bp->payload;
}

/*
 * mm_free - Free a block
 */
void mm_free(void *bp)
{
    assert(heap_listp != 0); // assert that mm_init was called
    if (bp == 0)
        return;

    /* Find block from user pointer */
    struct block *blk = bp - offsetof(struct block, payload);

    // Assertion 2 -- Check to ensure bp falls within the heap range
    assert(bp >= mem_heap_lo() && bp < mem_heap_hi());

    mark_block_free(blk, blk_size(blk));
    coalesce(blk);
}

/*
 * coalesce - Boundary tag coalescing. Return ptr to coalesced block
 */
static struct block *coalesce(struct block *bp)
{
    bool prev_alloc = prev_blk_footer(bp)->inuse; /* is previous block allocated? */
    bool next_alloc = !blk_free(next_blk(bp));    /* is next block allocated? */
    size_t size = blk_size(bp);

    if (prev_alloc && next_alloc)
    { /* Case 1 */
        // both are allocated, nothing to coalesce
        list_push_front(&free_lists[get_list_size(size)], &bp->elem);
        // return bp;
    }

    else if (prev_alloc && !next_alloc)
    { /* Case 2 */
        // Remove next block from the free list
        list_remove(&next_blk(bp)->elem);
        //  list_push_front(&free_list, &bp->elem);
        // combine this block and next block by extending it
        mark_block_free(bp, size + blk_size(next_blk(bp)));
        list_push_front(&free_lists[get_list_size(size + blk_size(next_blk(bp)))], &bp->elem);
    }

    else if (!prev_alloc && next_alloc)
    { /* Case 3 */
        // combine previous and this block by extending previous
        bp = prev_blk(bp);
        // Remove previous block from the free list
        list_remove(&bp->elem);
        // list_push_front(&free_list, &bp->elem);
        mark_block_free(bp, size + blk_size(bp));
        list_push_front(&free_lists[get_list_size(size + blk_size(bp))], &bp->elem);
    }

    else
    { /* Case 4 */

        // remove previous and next
        list_remove(&prev_blk(bp)->elem);
        list_remove(&next_blk(bp)->elem);
        // list_push_front(&free_list, &bp->elem);
        // combine all previous, this, and next block into one
        mark_block_free(prev_blk(bp),
                        size + blk_size(next_blk(bp)) + blk_size(prev_blk(bp)));
        bp = prev_blk(bp);
        list_push_front(&free_lists[get_list_size(size + blk_size(next_blk(bp)) + blk_size(prev_blk(bp)))], &bp->elem);
    }

    return bp;
}

/*
 * mm_realloc - Naive implementation of realloc
 */
void *mm_realloc(void *ptr, size_t size)
{
    /* If size == 0 then this is just free, and we return NULL. */
    if (size == 0)
    {
        mm_free(ptr);
        return 0;
    }

    /* If oldptr is NULL, then this is just malloc. */
    if (ptr == NULL)
    {
        return mm_malloc(size);
    }

    /* Copy the old data. */
    struct block *oldblock = ptr - offsetof(struct block, payload);
    size_t oldpayloadsize = blk_size(oldblock) * WSIZE - 2 * sizeof(struct boundary_tag);
    if (size < oldpayloadsize)
        oldpayloadsize = size;

    // From coalesce
    bool prev_alloc = prev_blk_footer(oldblock)->inuse; /* is previous block allocated? */
    bool next_alloc = !blk_free(next_blk(oldblock));    /* is next block allocated? */

    // From mm_malloc
    /* Adjust block size to include overhead and alignment reqs. */
    size_t bsize = align(size + 2 * sizeof(struct boundary_tag)); /* account for tags */
    if (bsize < size)
        return NULL; /* integer overflow */
    /* Adjusted block size in words */
    size_t awords = max(MIN_BLOCK_SIZE_WORDS, bsize / WSIZE); /* respect minimum size */

    // Initializing other variables
    struct block *prev_block = prev_blk(oldblock);
    struct block *next_block = next_blk(oldblock);
    size_t total_size_prev = blk_size(prev_block) + blk_size(oldblock);
    size_t total_size_next = blk_size(next_block) + blk_size(oldblock);

    // Case 0: If the requested size is less than the old block size, do nothing.
    if (awords <= blk_size(oldblock))
    {
        return oldblock->payload;
    }
    if (prev_alloc && get_footer(next_block)->inuse == -1) {
        //printf("%s\n", "yes");
        extend_heap(awords - blk_size(oldblock));
        next_block = next_blk(oldblock);
    }
    // Case 1: previous block free + next block allocated and og size is smaller than requested size
    if (!prev_alloc && next_alloc)
    {
        // Check if the combined size is sufficient
        if (total_size_prev >= awords)
        {
            // Remove the previous block from the free list
            assert(prev_block != NULL);
            list_remove(&prev_block->elem);

            // Mark as free and used
            mark_block_free(prev_block, total_size_prev);
            mark_block_used(prev_block, total_size_prev);

            // Move data to the beginning of the combined block
            memmove(prev_block->payload, oldblock->payload, oldpayloadsize);

            return prev_block->payload;
        }
    }

    // Case 2: previous block allocated + next block free and og size is smaller than requested size - currently doing both unallocated too
    else if (!next_alloc) 
    {
        if (total_size_next >= awords)
        {
            // Remove the next block from the free list
            assert(next_block != NULL);
            list_remove(&next_block->elem);

            // Mark as free and used
            mark_block_free(oldblock, total_size_next);
            mark_block_used(oldblock, total_size_next);

            // No memmove since we expand

            return oldblock->payload;
        }
    }

    void *newptr = mm_malloc(size);

    /* If realloc() fails the original block is left untouched  */
    if (!newptr)
    {
        return 0;
    }

    memcpy(newptr, ptr, oldpayloadsize);

    /* Free the old block. */
    mm_free(ptr);

    return newptr;
}

/*
 * checkheap - We don't check anything right now.
 */
void mm_checkheap(int verbose)
{
}

/*
 * The remaining routines are internal helper routines
 */

/*
 * extend_heap - Extend heap with free block and return its block pointer
 */
static struct block *extend_heap(size_t words)
{
    void *bp = mem_sbrk(words * WSIZE);

    if (bp == NULL)
        return NULL;

    /* Initialize free block header/footer and the epilogue header.
     * Note that we overwrite the previous epilogue here. */
    struct block *blk = bp - sizeof(FENCE);
    mark_block_free(blk, words);
    next_blk(blk)->header = FENCE;

    /* Coalesce if the previous block was free */
    return coalesce(blk);
}

/*
 * place - Place block of asize words at start of free block bp
 *         and split if remainder would be at least minimum block size
 */
static void place(struct block *bp, size_t asize)
{
    size_t csize = blk_size(bp);

    // Assertion 3 -- Check to ensure the block is free
    assert(blk_free(bp));

    // Assertion 4 -- Check to ensure the block size is greater than or equal to the requested size
    assert(asize <= csize);

    if ((csize - asize) >= MIN_BLOCK_SIZE_WORDS)
    {
        mark_block_used(bp, asize);
        list_remove(&bp->elem);
        bp = next_blk(bp);
        mark_block_free(bp, csize - asize);
        //  list_remove(&bp->elem);
        // Assertion 5 -- Check to ensure the freed block has a valid size
        assert(blk_size(bp) >= MIN_BLOCK_SIZE_WORDS);
        list_push_front(&free_lists[get_list_size(csize - asize)], &bp->elem);
    }
    else
    {
        mark_block_used(bp, csize);
        list_remove(&bp->elem);
    }
}

/*
 * find_fit - Find a fit for a block with asize words
 */
static struct block *find_fit(size_t asize)
{
    // Limit to prevent extensive search
    int search_limit = 5;
    for (int i = get_list_size(asize); i < NUM_FREE_LISTS; i++)
    {
        // Keep track of search depth
        int depth = 0;
        /* First fit search */
        for (struct list_elem *list = list_begin(&free_lists[i]); list != list_end(&free_lists[i]) && depth < search_limit; list = list_next(list), depth++)
        {
            struct block *bp = list_entry(list, struct block, elem);
            // Assertion 6 -- Check to ensure the block is free
            assert(blk_free(bp));
            if (asize <= blk_size(bp))
            {
                return bp;
            }
        }
    }
    return NULL; /* No fit */
}

/**
 * get_list_size - Calculates the index for the segregated free list
 * based on the size of the block
 *
 * Determines which free list to use by continously dividing the size
 * by 2 to categorize it into one of several size classes.
 *
 * @param size: block size in words
 * @return: index of the segregated free list that corresponds best
 *          compared to block size
 */
static int get_list_size(size_t size)
{
    // Round to minimum block size if it is lower than that
    if (size < MIN_BLOCK_SIZE_WORDS)
    {
        size = MIN_BLOCK_SIZE_WORDS;
    }

    int index = 0;
    while (size > 1 && index < NUM_FREE_LISTS - 1)
    {
        // Divide by 2 until right size is found
        size = size / 2;
        index++;
    }
    return index;
}

team_t team = {
    /* Team name */
    "Sample allocator using segregated list",
    /* First member's full name */
    "Nihar Satasia",
    "niharsatasia@vt.edu",
    /* Second member's full name (leave as empty strings if none) */
    "Patrick Walsh",
    "walsh968@vt.edu",
};
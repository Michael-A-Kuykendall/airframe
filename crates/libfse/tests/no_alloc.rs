use libfse::{FseMap, FseOpcode, FseScanner, Rule};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};

// A tracking allocator to detect heap activity in hot loops.
struct TrackingAllocator;

static ALLOC_COUNT: AtomicUsize = AtomicUsize::new(0);

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_COUNT.fetch_add(1, Ordering::SeqCst);
        System.alloc(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout)
    }
}

#[global_allocator]
static GLOBAL: TrackingAllocator = TrackingAllocator;

#[test]
fn test_zero_alloc_in_hot_loop() {
    // 1. Setup Phase (Allocations Allowed)
    let rules = vec![
        Rule::new("match_one", FseOpcode::Record(1)),
        Rule::new("match_two", FseOpcode::Record(2)),
        Rule::new("miss_me", FseOpcode::Record(3)),
    ];
    
    // Compile (creates DFA + Tables -> Allocs happen here)
    let map = FseMap::compile(rules).unwrap();
    
    // Initialize Scanner (creates BitVec -> Allocs happen here)
    let mut scanner = FseScanner::new(&map).unwrap();
    
    // Input Data (large enough to trigger reallocation if we were sloppy)
    let input = b"This string contains match_one and match_two but excludes the other one. ".repeat(100);

    // 2. Reset Counter
    ALLOC_COUNT.store(0, Ordering::SeqCst);

    // 3. Hot Loop
    let _ = scanner.scan(&input).unwrap();

    // 4. Assertion
    let count = ALLOC_COUNT.load(Ordering::SeqCst);
    
    // NOTE: In some environments, test harness itself allocates.
    // But our code should NOT. We check if count is suspiciously high.
    // A strict zero might flank on test infrastructure, but we aim for 0.
    // We print it to be sure.
    println!("Allocations during scan: {}", count);
    
    // If this fails, we are leaking/cloning somewhere in the loop.
    assert_eq!(count, 0, "Hot loop performed {} allocations! Expected 0.", count);
}

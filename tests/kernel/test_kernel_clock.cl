#define HILO_TO_ULONG(v) ((ulong)(v).x | ((ulong)(v).y << 32))
#define CLOCK_CHANGE_SAMPLE_COUNT 4096
#define REQUIRE(condition)                                                     \
  do {                                                                         \
    if (!(condition))                                                          \
      __builtin_trap();                                                        \
  } while (0)

#define CHECK_CLOCK_SCOPE(scope)                                               \
  do {                                                                         \
    ulong previous = clock_read_##scope();                                     \
    bool changed = false;                                                      \
    for (uint sample = 0; sample < CLOCK_CHANGE_SAMPLE_COUNT; ++sample) {      \
      ulong current = clock_read_##scope();                                    \
      REQUIRE(previous <= current);                                            \
      changed |= previous != current;                                          \
      previous = current;                                                      \
    }                                                                          \
    REQUIRE(changed);                                                          \
    ulong before = clock_read_##scope();                                       \
    uint2 split_value = clock_read_hilo_##scope();                             \
    ulong split = HILO_TO_ULONG(split_value);                                  \
    ulong after = clock_read_##scope();                                        \
    REQUIRE(before <= split && split <= after);                                \
  } while (0)

kernel void test_kernel_clock() {
  CHECK_CLOCK_SCOPE(device);
  CHECK_CLOCK_SCOPE(work_group);
  CHECK_CLOCK_SCOPE(sub_group);
}

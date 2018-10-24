#ifndef timing_h
#define timing_h

#define USE_WALL_TIME
#ifdef USE_WALL_TIME
#include <sys/time.h>
#else
#include <time.h>
#endif

// Handle for timing.
#ifdef USE_WALL_TIME
typedef struct timeval bench_timer_t;

/** Starts the clock for a benchmark. */
inline bench_timer_t time_start()
{
  bench_timer_t t;
  gettimeofday(&t, 0);
  return t;
}

/** Stops the clock and returns time elapsed in seconds.
 * Throws an error if time__start() was not called first.
 * */
inline double time_stop(bench_timer_t start)
{
  bench_timer_t end;
  bench_timer_t diff;
  gettimeofday(&end, 0);
  timersub(&end, &start, &diff);
  return (double)diff.tv_sec + ((double)diff.tv_usec / 1000000.0);
}

#else
typedef clock_t bench_timer_t;

inline bench_timer_t time_start()
{
  bench_timer_t t = clock();
  return t;
}

inline double time_stop(bench_timer_t start)
{
  clock_t _end = clock();
  return ((double)(_end - start)) / CLOCKS_PER_SEC;
}
#endif

#endif

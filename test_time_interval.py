import time
import statistics
import argparse
import ctypes

# 提升到 1ms 分辨率
ctypes.windll.winmm.timeBeginPeriod(1)
def measure_clock_stability(duration: float, interval: float):
    """
    测量 time.perf_counter() 调用间隔的统计信息。
    duration: 总测量时长（秒）
    interval: 采样间隔（秒），例如 0.005 表示 5ms
    """
    timestamps = []
    start = time.perf_counter()
    next_time = start

    # 循环直到达到测量时长
    while True:
        now = time.perf_counter()
        if now >= next_time:
            timestamps.append(now)
            next_time += interval
        else:
            # 等待到下一个采样点
            time.sleep(next_time - now)
        if now - start >= duration:
            break

    # 计算相邻采样的间隔
    intervals = [t2 - t1 for t1, t2 in zip(timestamps, timestamps[1:])]
    return intervals

def report_stats(intervals):
    mean = statistics.mean(intervals)
    stdev = statistics.stdev(intervals)
    minimum = min(intervals)
    maximum = max(intervals)
    print(f"样本数量: {len(intervals)}")
    print(f"平均间隔: {mean*1000:.3f} ms")
    print(f"标准差  : {stdev*1000:.3f} ms")
    print(f"最小间隔: {minimum*1000:.3f} ms")
    print(f"最大间隔: {maximum*1000:.3f} ms")

def main():
    parser = argparse.ArgumentParser(description="检查系统时钟频率稳定性")
    parser.add_argument(
        "-d", "--duration", type=float, default=5.0,
        help="测量总时长，单位秒 (默认 5s)"
    )
    parser.add_argument(
        "-i", "--interval", type=float, default=0.005,
        help="采样间隔，单位秒 (默认 0.005 = 5ms)"
    )
    args = parser.parse_args()

    print(f"开始测量: 时长 {args.duration}s, 采样间隔 {args.interval*1000:.1f}ms")
    intervals = measure_clock_stability(args.duration, args.interval)
    report_stats(intervals)

if __name__ == "__main__":
    main()
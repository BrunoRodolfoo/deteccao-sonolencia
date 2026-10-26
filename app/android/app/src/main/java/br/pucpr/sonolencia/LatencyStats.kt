package br.pucpr.sonolencia

/**
 * Estatísticas de latência sobre as últimas [capacity] amostras (buffer circular):
 * média, p50 e p95 (percentil por interpolação linear, como numpy.percentile).
 */
class LatencyStats(private val capacity: Int = 300) {
    private val buf = FloatArray(capacity)
    private var size = 0
    private var next = 0
    var last = 0f
        private set
    var total = 0L
        private set

    fun add(ms: Float) {
        last = ms
        total++
        buf[next] = ms
        next = (next + 1) % capacity
        if (size < capacity) size++
    }

    val count: Int get() = size

    fun mean(): Float {
        if (size == 0) return 0f
        var s = 0f
        for (i in 0 until size) s += buf[i]
        return s / size
    }

    fun percentile(p: Float): Float {
        if (size == 0) return 0f
        val sorted = buf.copyOf(size)
        sorted.sort()
        val rank = (p / 100f) * (size - 1)
        val lo = kotlin.math.floor(rank).toInt()
        val hi = minOf(lo + 1, size - 1)
        val frac = rank - lo
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }

    fun p50() = percentile(50f)
    fun p95() = percentile(95f)

    fun reset() {
        size = 0
        next = 0
        last = 0f
        total = 0
    }
}

package br.pucpr.sonolencia

import org.junit.Assert.assertEquals
import org.junit.Test

class PerclosAndLatencyTest {
    @Test
    fun perclosFracaoNaJanela() {
        val p = PerclosWindow(windowMs = 1000L)
        for (t in 0 until 10) p.update(t * 100L, closed = t < 3)   // 3 fechados em 10
        assertEquals(0.3f, p.value, 1e-6f)
        assertEquals(900L, p.coverageMs)
        // Avança 2 s: janela descarta tudo que tem mais de 1 s
        p.update(2500L, false)
        assertEquals(0f, p.value, 1e-6f)
    }

    @Test
    fun latencyPercentisComoNumpy() {
        val s = LatencyStats(10)
        listOf(10f, 20f, 30f, 40f, 50f).forEach { s.add(it) }
        assertEquals(30f, s.mean(), 1e-6f)
        assertEquals(30f, s.p50(), 1e-6f)
        assertEquals(48f, s.p95(), 1e-5f)   // numpy.percentile([10..50], 95) = 48
        assertEquals(50f, s.last, 0f)
    }

    @Test
    fun latencyBufferCircular() {
        val s = LatencyStats(3)
        listOf(100f, 1f, 2f, 3f).forEach { s.add(it) }   // 100 sai do buffer
        assertEquals(3, s.count)
        assertEquals(2f, s.mean(), 1e-6f)
        assertEquals(4L, s.total)
    }
}

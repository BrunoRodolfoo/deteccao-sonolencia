package br.pucpr.sonolencia

import java.util.ArrayDeque

/**
 * PERCLOS em janela deslizante de [windowMs] milissegundos: fração dos quadros da janela
 * em que o olho foi considerado fechado (EAR abaixo do limiar). Valores sustentados acima
 * de ~0.15 indicam sonolência na literatura (Wierwille et al., 1994).
 */
class PerclosWindow(private val windowMs: Long = 60_000L) {
    private class Event(val t: Long, val closed: Boolean)

    private val events = ArrayDeque<Event>()
    private var closedCount = 0

    /** Duração coberta pela janela até agora (ms); útil para exigir cobertura mínima. */
    val coverageMs: Long get() = if (events.isEmpty()) 0L else events.last.t - events.first.t

    fun update(timeMs: Long, closed: Boolean): Float {
        events.addLast(Event(timeMs, closed))
        if (closed) closedCount++
        while (events.isNotEmpty() && timeMs - events.first.t > windowMs) {
            val e = events.removeFirst()
            if (e.closed) closedCount--
        }
        return value
    }

    val value: Float
        get() = if (events.isEmpty()) 0f else closedCount.toFloat() / events.size

    fun reset() {
        events.clear()
        closedCount = 0
    }
}

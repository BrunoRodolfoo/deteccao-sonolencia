package br.pucpr.sonolencia

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View

/** Borda vermelha piscante sobre a pré-visualização quando há alerta. */
class AlertOverlayView @JvmOverloads constructor(
    context: Context, attrs: AttributeSet? = null, defStyleAttr: Int = 0,
) : View(context, attrs, defStyleAttr) {

    private val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
        color = Color.RED
        style = Paint.Style.STROKE
        strokeWidth = 32f
    }
    private var visibleStroke = true

    var active: Boolean = false
        set(value) {
            if (field != value) {
                field = value
                visibleStroke = true
                invalidate()
            }
        }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        if (!active) return
        if (visibleStroke) {
            val half = paint.strokeWidth / 2f
            canvas.drawRect(half, half, width - half, height - half, paint)
        }
        visibleStroke = !visibleStroke
        postInvalidateDelayed(350L)
    }
}

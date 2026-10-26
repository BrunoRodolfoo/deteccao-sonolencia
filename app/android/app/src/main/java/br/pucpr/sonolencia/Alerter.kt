package br.pucpr.sonolencia

import android.content.Context
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.SystemClock
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log

/** Beep (ToneGenerator) + vibração, com limitação de taxa para não disparar a cada quadro. */
class Alerter(context: Context, private val minIntervalMs: Long = 2000L) {
    private var tone: ToneGenerator? = try {
        ToneGenerator(AudioManager.STREAM_ALARM, 100)
    } catch (e: RuntimeException) {
        Log.w("Alerter", "ToneGenerator indisponível", e)
        null
    }

    private val vibrator: Vibrator? = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
        (context.getSystemService(Context.VIBRATOR_MANAGER_SERVICE) as? VibratorManager)?.defaultVibrator
    } else {
        @Suppress("DEPRECATION")
        context.getSystemService(Context.VIBRATOR_SERVICE) as? Vibrator
    }

    private var lastAlertMs = 0L

    /** Dispara beep + vibração se passou [minIntervalMs] desde o último alerta. */
    fun alert() {
        val now = SystemClock.uptimeMillis()
        if (now - lastAlertMs < minIntervalMs) return
        lastAlertMs = now
        try {
            tone?.startTone(ToneGenerator.TONE_CDMA_ALERT_CALL_GUARD, 500)
        } catch (e: RuntimeException) {
            Log.w("Alerter", "Falha no beep", e)
        }
        try {
            vibrator?.vibrate(VibrationEffect.createOneShot(400L, VibrationEffect.DEFAULT_AMPLITUDE))
        } catch (e: RuntimeException) {
            Log.w("Alerter", "Falha na vibração", e)
        }
    }

    fun release() {
        tone?.release()
        tone = null
    }
}

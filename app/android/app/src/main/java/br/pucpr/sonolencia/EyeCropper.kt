package br.pucpr.sonolencia

import android.graphics.Bitmap
import kotlin.math.roundToInt

/**
 * Recorte do olho para a CNN: caixa quadrada em torno do centro do olho (lado = 1.6 x
 * distância entre os cantos p1-p4), convertida para cinza e redimensionada para
 * [outW] x [outH]. Nada é gravado em disco.
 *
 * Equivalência com o treino (src/pipeline_b/datasets.py): as imagens do MRL já são
 * recortes do olho e o recorte "crop" do CEW é uma caixa quadrada de 0,35 x largura da
 * face centrada no olho; para uma face de 100 px isso dá 36 px, ou seja, ~1,6 x a largura
 * do olho (~22 px). A caixa é deslocada (e não cortada) quando encosta na borda, para
 * continuar quadrada; o redimensionamento usa filtro bilinear (o treino usou INTER_AREA).
 */
object EyeCropper {
    /** Retorna os pixels em cinza (0..255) row-major, ou null se a caixa for degenerada. */
    fun cropGray(src: Bitmap, box: FaceMetrics.EyeBox, outW: Int, outH: Int): FloatArray? {
        var side = box.side.roundToInt()
        if (side < 4) return null
        side = minOf(side, src.width, src.height)
        // Mantém a caixa quadrada: desloca para dentro da imagem em vez de cortá-la.
        val x0 = (box.cx - side / 2f).roundToInt().coerceIn(0, src.width - side)
        val y0 = (box.cy - side / 2f).roundToInt().coerceIn(0, src.height - side)
        val crop = Bitmap.createBitmap(src, x0, y0, side, side)
        val scaled = if (crop.width == outW && crop.height == outH) crop
        else Bitmap.createScaledBitmap(crop, outW, outH, true)
        val px = IntArray(outW * outH)
        scaled.getPixels(px, 0, outW, 0, 0, outW, outH)
        val gray = FloatArray(outW * outH)
        for (i in px.indices) {
            val c = px[i]
            val r = (c shr 16) and 0xFF
            val g = (c shr 8) and 0xFF
            val b = c and 0xFF
            gray[i] = 0.299f * r + 0.587f * g + 0.114f * b   // mesma conversão do cv2.COLOR_RGB2GRAY
        }
        // createBitmap devolve o PRÓPRIO bitmap de origem quando o recorte cobre a imagem
        // inteira: nunca reciclar o quadro que ainda pertence ao pipeline.
        if (scaled !== crop && scaled !== src) scaled.recycle()
        if (crop !== src) crop.recycle()
        return gray
    }
}

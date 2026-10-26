package br.pucpr.sonolencia

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject

/**
 * Metadados opcionais dos modelos (`<nome>.json` ao lado do `.tflite` em assets), gerados
 * pelos scripts de exportação em `src/`. Todos os campos têm padrão seguro; o formato de
 * entrada (shape/dtype/quantização) é lido diretamente do interpretador TFLite, então o
 * JSON só ajusta a normalização e os nomes das classes.
 *
 * Campos reconhecidos (todos opcionais; chave em inglês ou em português):
 *  - "input_range"/"faixa_entrada": "0-1" (padrão) ou "0-255"
 *  - "mean"/"media" e "std"/"desvio": ESCALARES de normalização (x - mean)/std aplicados
 *    após o range (vetores, como o "mean" da camada Normalization embutida no modelo
 *    temporal, são ignorados: essa normalização já está dentro do .tflite)
 *  - "class_names"/"classes": lista (ou objeto {"0": "alerta", ...}) com os nomes das saídas;
 *    usada para achar o índice de "open"/"aberto"
 *  - "open_index"/"indice_aberto": índice explícito da classe "olho aberto" (padrão 1:
 *    0 fechado, 1 aberto, convenção do MRL Eye Dataset)
 *  - "feature_order"/"ordem_caracteristicas": ordem das 4 características (esperado
 *    frequência, amplitude, duração, velocidade; se diferir, o app avisa no log)
 */
data class ModelMetadata(
    val inputRange255: Boolean = false,
    val mean: Float = 0f,
    val std: Float = 1f,
    val classNames: List<String> = emptyList(),
    val openIndex: Int = 1,
    val featureOrder: List<String> = emptyList(),
) {
    companion object {
        fun load(context: Context, assetName: String): ModelMetadata {
            val text = try {
                context.assets.open(assetName).bufferedReader().use { it.readText() }
            } catch (e: Exception) {
                return ModelMetadata()
            }
            return try {
                parse(JSONObject(text))
            } catch (e: Exception) {
                ModelMetadata()
            }
        }

        /** Aceita chaves em inglês e as chaves em português usadas por models/*.json. */
        fun parse(o: JSONObject): ModelMetadata {
            val range = firstString(o, "input_range", "faixa_entrada") ?: "0-1"
            val names = ArrayList<String>()
            val classes = o.opt("class_names") ?: o.opt("classes")
            when (classes) {
                is JSONArray -> for (i in 0 until classes.length()) names.add(classes.optString(i))
                is JSONObject -> {
                    // {"0": "alerta", "1": ...} -> ordena pelo índice numérico
                    val keys = classes.keys().asSequence().toList().sortedBy { it.toIntOrNull() ?: Int.MAX_VALUE }
                    for (k in keys) names.add(classes.optString(k))
                }
                else -> Unit
            }
            val order = ArrayList<String>()
            (o.optJSONArray("feature_order") ?: o.optJSONArray("ordem_caracteristicas"))?.let { arr ->
                for (i in 0 until arr.length()) order.add(arr.optString(i))
            }
            var openIdx = if (o.has("open_index")) o.optInt("open_index", -1) else o.optInt("indice_aberto", -1)
            if (openIdx < 0) {
                val found = names.indexOfFirst {
                    val n = it.lowercase()
                    n.contains("open") || n.contains("abert")
                }
                openIdx = if (found >= 0) found else 1
            }
            val mean = firstDouble(o, "mean", "media") ?: 0.0
            val std = (firstDouble(o, "std", "desvio") ?: 1.0).let { if (it == 0.0) 1.0 else it }
            return ModelMetadata(
                inputRange255 = range.replace(" ", "") == "0-255",
                mean = mean.toFloat(),
                std = std.toFloat(),
                classNames = names,
                openIndex = openIdx,
                featureOrder = order,
            )
        }

        private fun firstString(o: JSONObject, vararg keys: String): String? =
            keys.firstOrNull { o.has(it) && o.opt(it) is String }?.let { o.optString(it) }

        /** Só aceita valores escalares (a chave "mean" do modelo temporal é um vetor e é ignorada). */
        private fun firstDouble(o: JSONObject, vararg keys: String): Double? =
            keys.firstOrNull { o.has(it) && o.opt(it) is Number }?.let { o.optDouble(it) }
    }
}

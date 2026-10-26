"""Modelos para classificação de estado ocular (aberto/fechado).

EyeNet: CNN compacta (entrada 48x48x1, saída sigmoide P(aberto)), pensada
para rodar em TFLite int8 no celular. Três blocos Conv-BN-ReLU-MaxPool
(16-32-64 filtros), GAP, Dense 32, Dropout 0.3, Dense 1 sigmoide.
Aproximadamente 30 mil parâmetros com os filtros padrão (16/32/64) e
~100 mil com (24/48/96) — o padrão do treino usa (16, 32, 64) + uma conv
extra de 64 no último bloco, ficando na faixa de 50-70 mil parâmetros.

MobileNetV3-Small (keras.applications) fica documentada como alternativa
opcional; só é treinada se `--mobilenet` for passado no treino.
"""

from __future__ import annotations

import keras
from keras import layers

IMG_SIZE = 48
# Momento das médias móveis da BatchNormalization. O pipeline usa 0,9 em vez do
# padrão do Keras (0,99). A motivação original (modelo prevendo classe constante
# em modo de inferência com 0,99) foi observada apenas em execuções curtas de
# teste (--smoke, ~50 passos) e NÃO se reproduz no treino completo: a ablação
# em src/pipeline_b/ablation_bn_momentum.py (results/exp3_bn_momentum_ablation.csv)
# mostra que, no fold 1 do protocolo A, 0,99 e 0,9 dão resultados comparáveis
# (teste: 93,2% vs 91,2%). O valor 0,9 é mantido porque é o usado em todos os
# resultados reportados; não é uma escolha decisiva.
BN_MOMENTUM = 0.9


def build_eyenet(filters=(16, 32, 64), dense=32, dropout=0.3, name="EyeNet") -> keras.Model:
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1), name="olho_48x48")
    x = inp
    for i, f in enumerate(filters):
        x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i + 1}a")(x)
        x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn{i + 1}a")(x)
        x = layers.ReLU(name=f"relu{i + 1}a")(x)
        if i == len(filters) - 1:  # bloco final com uma conv extra
            x = layers.Conv2D(f, 3, padding="same", use_bias=False, name=f"conv{i + 1}b")(x)
            x = layers.BatchNormalization(momentum=BN_MOMENTUM, name=f"bn{i + 1}b")(x)
            x = layers.ReLU(name=f"relu{i + 1}b")(x)
        x = layers.MaxPooling2D(2, name=f"pool{i + 1}")(x)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(dense, activation="relu", name="fc")(x)
    x = layers.Dropout(dropout, name="drop")(x)
    out = layers.Dense(1, activation="sigmoid", name="p_aberto")(x)
    return keras.Model(inp, out, name=name)


def build_mobilenetv3(alpha=0.75) -> keras.Model:
    """MobileNetV3-Small com pesos ImageNet; entrada 96x96x3 (cinza replicado)."""
    from keras.applications import MobileNetV3Small
    inp = layers.Input((IMG_SIZE, IMG_SIZE, 1))
    x = layers.Resizing(96, 96)(inp)
    x = layers.Concatenate()([x, x, x])
    x = layers.Rescaling(255.0)(x)  # MobileNetV3 do Keras espera [0,255]
    base = MobileNetV3Small(input_shape=(96, 96, 3), alpha=alpha, include_top=False,
                            weights="imagenet", include_preprocessing=True, pooling="avg")
    x = base(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return keras.Model(inp, out, name="MobileNetV3Small")


def build_augmentation() -> keras.Sequential:
    """Augmentation leve, aplicada só no treino (documentada no results/exp3_eye_state.md):
    flip horizontal, rotação até ±10°, translação até ±8%, zoom ±10%,
    brilho ±15% e contraste ±20%."""
    return keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(10 / 360, fill_mode="nearest"),
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        layers.RandomZoom(0.1, fill_mode="nearest"),
        layers.RandomBrightness(0.15, value_range=(0, 1)),
        layers.RandomContrast(0.2),
    ], name="augmentation")


if __name__ == "__main__":
    m = build_eyenet()
    m.summary()
    print("parâmetros:", m.count_params())

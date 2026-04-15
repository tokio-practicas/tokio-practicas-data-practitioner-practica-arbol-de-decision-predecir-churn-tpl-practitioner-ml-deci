# Práctica: Modelo de Árbol de Decisión para Predecir Churn

## Contexto

Trabajas como junior Data Analyst en una empresa de telecomunicaciones. El equipo de retención quiere predecir qué clientes van a cancelar su servicio (churn) para poder actuar antes. Tu tarea es entrenar un modelo básico de Machine Learning y explicar los resultados.

## Dataset

`data/telecom_churn.csv` — 1.000 clientes con las siguientes variables:

| Variable | Descripción |
|----------|-------------|
| `customer_id` | Identificador único |
| `tenure_months` | Meses como cliente (1-72) |
| `monthly_charges` | Cargo mensual (€) |
| `total_charges` | Cargo total acumulado (€) |
| `contract_type` | month-to-month / one_year / two_year |
| `internet_service` | fiber / dsl / none |
| `online_security` | yes / no |
| `tech_support` | yes / no |
| `payment_method` | electronic_check / bank_transfer / credit_card / mailed_check |
| `senior_citizen` | 0 / 1 |
| `num_support_tickets` | Tickets de soporte abiertos (0-15) |
| `churn` | **Target** — 0 (se queda) / 1 (cancela) |

## Tareas

Completa el notebook `notebook/practica_churn.ipynb` siguiendo las secciones:

1. **Carga de datos** — Lee el CSV con pandas
2. **Exploración rápida** — `.shape`, `.dtypes`, `.describe()`, distribución del target
3. **Preparación** — Codifica variables categóricas, haz train/test split (80/20, `random_state=42`)
4. **Entrenamiento** — Entrena un `DecisionTreeClassifier` con `random_state=42`
5. **Evaluación** — Calcula accuracy, precision y recall. Muestra la confusion matrix
6. **Top 3 features** — Usa `feature_importances_` para identificar las 3 variables más importantes. Gráfico de barras + explicación en markdown
7. **Conclusiones** — Interpreta los resultados: ¿tiene sentido lo que dice el modelo? ¿Qué recomendarías al equipo de retención?

## Cómo entregar

Esta práctica usa un flujo basado en **Pull Requests**. No puedes hacer push directo a `main`.

1. **Crea una rama de entrega**:
   ```bash
   git checkout -b entrega
   ```
2. **Completa la práctica y commitea**:
   ```bash
   git add .
   git commit -m "Mi entrega"
   ```
3. **Sube la rama**:
   ```bash
   git push -u origin entrega
   ```
4. **Abre un Pull Request** desde la web de GitHub (`entrega` → `main`).
5. **Recibirás feedback automáticamente** en el PR:
   - Resultados de los tests (pytest)
   - Revisión del código con IA (Claude)
6. **Itera**: cada nuevo commit que hagas en la rama `entrega` dispara una nueva revisión y actualiza el PR.
7. **Cerrar entrega**: haz merge del PR a `main` cuando los tests pasen y estés contento con el feedback. Solo podrás mergear si los tests pasan.

## Evaluación

| Criterio | Peso |
|----------|------|
| Notebook ejecuta sin errores | 20% |
| Preparación correcta (encoding, split) | 20% |
| Modelo entrenado y evaluado | 20% |
| Feature importance identificada y explicada | 20% |
| Calidad del código y conclusiones | 20% |

## Requisitos técnicos

```bash
pip install -r requirements.txt
```

**Tiempo estimado:** 3-4 horas

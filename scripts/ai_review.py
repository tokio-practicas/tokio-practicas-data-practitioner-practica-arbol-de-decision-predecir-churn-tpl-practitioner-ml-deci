"""Revisión automática con IA usando Claude API."""
import os
import json
import glob


def read_notebook():
    notebooks = glob.glob("notebook/*.ipynb")
    if not notebooks:
        return None
    with open(notebooks[0]) as f:
        nb = json.load(f)
    cells_text = []
    for i, cell in enumerate(nb["cells"]):
        cell_type = cell["cell_type"]
        source = "".join(cell.get("source", []))
        if source.strip():
            cells_text.append(f"[Celda {i + 1} ({cell_type})]:\n{source}")
    return "\n\n".join(cells_text)


def read_test_results():
    try:
        with open("test_output.txt") as f:
            return f.read()
    except FileNotFoundError:
        return "No hay resultados de tests disponibles"


def get_ai_feedback(notebook_content, test_results):
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return "⚠️ No se configuró ANTHROPIC_API_KEY. Contacta al profesor."

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)

        prompt = f"""Eres un profesor de Data Science revisando la práctica de un alumno.
La práctica consiste en entrenar un árbol de decisión para predecir churn de clientes
de telecomunicaciones.

Resultados de los tests automáticos:
{test_results}

Contenido del notebook del alumno:
{notebook_content}

Evalúa el trabajo en estos criterios (1-10 cada uno):

1. **Carga y exploración de datos** — ¿Miró shape, tipos, nulos, distribución del target?
2. **Preparación** — ¿Codificó variables categóricas correctamente? ¿Split adecuado?
3. **Modelo** — ¿Entrenó correctamente? ¿Usó random_state?
4. **Evaluación** — ¿Calculó accuracy, precision, recall? ¿Interpretó la confusion matrix?
5. **Feature importance** — ¿Identificó las top 3? ¿Las explicó con sentido?
6. **Calidad del código** — ¿Es limpio, comentado, reproducible?

Da feedback constructivo y específico en español. Sé amable pero riguroso.
Termina con:
- **Nota global**: Suspenso / Aprobado / Notable / Sobresaliente
- **Consejo de mejora**: Una cosa concreta que mejoraría más su trabajo

Formato: Markdown."""

        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text
    except Exception as e:
        return f"⚠️ Error en revisión IA: {str(e)}"


def main():
    notebook_content = read_notebook()
    if not notebook_content:
        feedback = "⚠️ No se encontró el notebook de la práctica."
    else:
        test_results = read_test_results()
        feedback = get_ai_feedback(notebook_content, test_results)

    with open("ai_feedback.md", "w") as f:
        f.write(feedback)
    print("AI feedback generated successfully")


if __name__ == "__main__":
    main()

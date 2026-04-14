"""Tests automáticos para la práctica de árbol de decisión."""
import pytest
import subprocess
import json
import os

ROOT = os.path.dirname(os.path.dirname(__file__))


def run_notebook():
    """Ejecuta el notebook y devuelve el resultado."""
    result = subprocess.run(
        [
            "jupyter", "nbconvert",
            "--to", "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=120",
            os.path.join(ROOT, "notebook", "practica_churn.ipynb"),
            "--output", "executed.ipynb",
        ],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    return result


def get_notebook():
    """Lee el notebook ejecutado."""
    path = os.path.join(ROOT, "notebook", "executed.ipynb")
    with open(path) as f:
        return json.load(f)


def all_code_source(nb):
    """Concatena todo el código del notebook."""
    return "\n".join(
        "".join(c["source"])
        for c in nb["cells"]
        if c["cell_type"] == "code"
    )


class TestNotebookExecution:
    @pytest.fixture(autouse=True, scope="class")
    def execute_notebook(self):
        result = run_notebook()
        assert result.returncode == 0, f"El notebook no ejecuta: {result.stderr[:500]}"

    def test_no_cell_errors(self):
        nb = get_notebook()
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                for output in cell.get("outputs", []):
                    assert output.get("output_type") != "error", (
                        f"Error en celda: {output.get('evalue', 'desconocido')}"
                    )

    def test_data_loaded(self):
        src = all_code_source(get_notebook())
        assert "read_csv" in src or "telecom_churn" in src, (
            "No se encontró carga del dataset"
        )

    def test_train_test_split_used(self):
        src = all_code_source(get_notebook())
        assert "train_test_split" in src, "No se usó train_test_split"

    def test_decision_tree_used(self):
        src = all_code_source(get_notebook())
        assert "DecisionTreeClassifier" in src, "No se usó DecisionTreeClassifier"

    def test_evaluation_metrics(self):
        src = all_code_source(get_notebook()).lower()
        assert "accuracy" in src, "Falta métrica: accuracy"
        assert "precision" in src, "Falta métrica: precision"
        assert "recall" in src, "Falta métrica: recall"

    def test_feature_importances(self):
        src = all_code_source(get_notebook())
        assert "feature_importances_" in src, "No se usó feature_importances_"

    def test_conclusions_written(self):
        nb = get_notebook()
        md = "\n".join(
            "".join(c["source"])
            for c in nb["cells"]
            if c["cell_type"] == "markdown"
        )
        # Check student wrote something beyond the template
        assert md.count("*") < md.count(" ") // 5, (
            "Parece que no se completaron las conclusiones"
        )

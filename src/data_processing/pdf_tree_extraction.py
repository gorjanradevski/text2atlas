import pdftotree


def get_pdf_tree(pdf_path: str, html_path: str) -> dict:
    return pdftotree.parse(
        pdf_path,
        html_path=html_path,
        model_type=None,
        model_path=None,
        favor_figures=True
    )
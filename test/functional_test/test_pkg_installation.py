def test_pkg_installation():
    try:
        import mypackage
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."
        
def test_ingestion():
    try:
        from mypackage import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."

def test_train():
    try:
        from mypackage import train
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."

def test_score():
    try:
        from mypackage import score
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."
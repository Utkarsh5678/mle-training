def test_pkg_installation():
    try:
        import housingpriceprediction
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."
        
def test_ingestion():
    try:
        from housingpriceprediction import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."

def test_train():
    try:
        from  housingpriceprediction import train
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."

def test_score():
    try:
        from housingpriceprediction import score
    except Exception as e:
        assert False, f"Error: {e}. mypackage is not installed properly."
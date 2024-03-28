def ingest1():
    try:
        from awesome_package import models
    except Exception as e:
        assert False, f"Error: {e}. ingest_data is not installed correctly."
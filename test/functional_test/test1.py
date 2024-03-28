def ingest():
    try:
        from awesome_package import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. ingest_data is not installed correctly."
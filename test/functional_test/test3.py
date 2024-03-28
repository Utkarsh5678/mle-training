def scores():
    try:
        from awesome_package import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. score is not installed correctly."
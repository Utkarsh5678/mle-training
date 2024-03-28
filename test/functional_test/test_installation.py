def test_pkg_installation():
    try:
        import awesome_package
    except Exception as e:
        assert False, f"Error: {e}. Awesome_package is not installed correctly."    
def ingest():
    try:
        from awesome_package import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. ingest_data is not installed correctly."
def model():
    try:
        from awesome_package import models
    except Exception as e:
        assert False, f"Error: {e}. models is not installed correctly."
def scores():
    try:
        from awesome_package import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. score is not installed correctly."
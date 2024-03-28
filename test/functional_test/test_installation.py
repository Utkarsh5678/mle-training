def test_pkg_installation():
    try:
        import models
    except Exception as e:
        assert False, f"Error: {e}. Awesome_package is not installed correctly."

    try:
        from models import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. ingest_data is not installed correctly."

    try:
        from models import methods
    except Exception as e:
        assert False, f"Error: {e}. models is not installed correctly."
    try:
        from models import score
    except Exception as e:
        assert False, f"Error: {e}. score is not installed correctly."


    

    
        
    # with pytest.raises(ImportError):
    #     import Awesome_package
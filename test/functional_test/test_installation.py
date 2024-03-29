def test_pkg_installation():
    try:
        import mypackage
    except Exception as e:
        assert False, f"Error: {e}. Awesome_package is not installed correctly."

    try:
        from mypackage import ingest_data
    except Exception as e:
        assert False, f"Error: {e}. ingest_data is not installed correctly."

    try:
        from mypackage import methods
    except Exception as e:
        assert False, f"Error: {e}. methods is not installed correctly."
    try:
        from mypackage import score
    except Exception as e:
        assert False, f"Error: {e}. score is not installed correctly."


    

    
        
    # with pytest.raises(ImportError):
    #     import Awesome_package
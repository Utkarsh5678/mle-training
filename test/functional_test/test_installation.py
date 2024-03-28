def test_pkg_installation():
    try:
        import awesome_package
    except Exception as e:
        assert False, f"Error: {e}. Awesome_package is not installed correctly."

    
        
    # with pytest.raises(ImportError):
    #     import Awesome_package
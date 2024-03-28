def model():
    try:
        from awesome_package import models
    except Exception as e:
        assert False, f"Error: {e}. models is not installed correctly."
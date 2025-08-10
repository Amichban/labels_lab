"""
Example integration test file for Claude Native Template.
Replace with your actual integration tests.
"""

def test_integration_example():
    """Example integration test."""
    # Simulate an integration test
    components = ["frontend", "backend", "database"]
    system = {"components": components}
    
    assert len(system["components"]) == 3
    assert "backend" in system["components"]
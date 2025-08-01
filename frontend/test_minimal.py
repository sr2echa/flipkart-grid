#!/usr/bin/env python3
"""
Test script to check if the minimal autosuggest system works
"""

import sys
import os

# Add current directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

def test_imports():
    """Test if all imports work."""
    print("🧪 Testing imports...")
    
    try:
        from simple_autosuggest import SimpleAutosuggestSystem
        print("✅ SimpleAutosuggestSystem imported successfully")
    except Exception as e:
        print(f"❌ Failed to import SimpleAutosuggestSystem: {e}")
        return False
    
    try:
        from data_preprocessing import DataPreprocessor
        print("✅ DataPreprocessor imported successfully")
    except Exception as e:
        print(f"❌ Failed to import DataPreprocessor: {e}")
        return False
    
    return True

def test_data_preprocessing():
    """Test if data preprocessing works."""
    print("\n🧪 Testing data preprocessing...")
    
    try:
        from data_preprocessing import DataPreprocessor
        preprocessor = DataPreprocessor()
        preprocessor.run_all_preprocessing()
        data = preprocessor.get_processed_data()
        print(f"✅ Data preprocessing successful. Keys: {list(data.keys())}")
        return True, data
    except Exception as e:
        print(f"❌ Data preprocessing failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None

def test_autosuggest_system(data):
    """Test if autosuggest system works."""
    print("\n🧪 Testing autosuggest system...")
    
    try:
        from simple_autosuggest import SimpleAutosuggestSystem
        autosuggest = SimpleAutosuggestSystem()
        autosuggest.build_system(data)
        
        # Test suggestions
        suggestions = autosuggest.get_suggestions("lap", max_suggestions=5)
        print(f"✅ Autosuggest system works! Sample suggestions for 'lap': {suggestions}")
        return True
    except Exception as e:
        print(f"❌ Autosuggest system failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🎯 Testing Minimal Autosuggest System")
    print("=" * 50)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed")
        return False
    
    # Test data preprocessing
    success, data = test_data_preprocessing()
    if not success:
        print("\n❌ Data preprocessing tests failed")
        return False
    
    # Test autosuggest system
    if not test_autosuggest_system(data):
        print("\n❌ Autosuggest system tests failed")
        return False
    
    print("\n🎉 All tests passed!")
    print("✅ The minimal autosuggest system is working correctly")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Integration test for plot_composition_temperature function
"""
import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcp_materials_project.handlers.calphad.phase_diagrams.phase_diagrams import CalPhadHandler

async def test_plot_composition_temperature():
    """Test the complete plot_composition_temperature workflow"""
    
    print("🧪 Testing plot_composition_temperature function\n")
    
    # Create handler
    handler = CalPhadHandler()
    print(f"✓ Handler created, TDB directory: {handler.tdb_dir}")
    
    # Test composition: Al20Zn80
    composition = "Al20Zn80"
    print(f"\n📊 Generating plot for composition: {composition}")
    
    try:
        # Call the function
        result = await handler.plot_composition_temperature(
            composition=composition,
            min_temperature=None,  # Use auto range
            max_temperature=None,
            interactive="html"
        )
        
        print(f"\n✓ Function completed!")
        print(f"Result: {result[:200]}...")
        
        # Check if metadata was created
        if hasattr(handler, '_last_image_metadata'):
            metadata = handler._last_image_metadata
            print(f"\n✓ Metadata created:")
            print(f"  Composition: {metadata.get('composition')}")
            print(f"  System: {metadata.get('system')}")
            print(f"  Temp range: {metadata.get('temperature_range_K')}")
            print(f"  Interactive: {metadata.get('interactive')}")
            
            if 'image_info' in metadata:
                img_info = metadata['image_info']
                print(f"  Image format: {img_info.get('format')}")
                print(f"  Image URL: {img_info.get('url')}")
                print(f"  HTML URL: {img_info.get('interactive_html_url')}")
            
            if 'analysis' in metadata:
                analysis = metadata['analysis']
                print(f"\n✓ Analysis generated ({len(analysis)} chars):")
                print(f"  {analysis[:200]}...")
        else:
            print("\n❌ No metadata found!")
        
        # Check if image URL was set
        if hasattr(handler, '_last_image_url'):
            print(f"\n✓ Image URL set: {handler._last_image_url}")
        else:
            print("\n❌ No image URL found!")
        
        # Check if HTML URL was set
        if hasattr(handler, '_last_html_url'):
            print(f"✓ HTML URL set: {handler._last_html_url}")
        else:
            print("\n⚠️  No HTML URL found (might be using matplotlib fallback)")
        
        print("\n✅ Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(test_plot_composition_temperature())
    sys.exit(0 if success else 1)


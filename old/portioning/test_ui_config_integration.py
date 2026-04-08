"""Tests for ui.py integration with config_manager."""

import unittest
from unittest.mock import patch, MagicMock
from old.portioning.config_manager import AppConfig


class TestUiConfigIntegration(unittest.TestCase):
    """Test that ui.py correctly uses config_manager for default values."""
    
    def test_sidebar_controls_uses_config_values(self):
        """Test that sidebar_controls loads and uses config values for pieces_per_min and line_eff."""
        # Create a mock config with specific values
        mock_config = AppConfig(
            buckets=[(0, 324)],
            illegal_pairs={"C": ["D"]},
            trim_cap=15,
            time_limit_sec=60,
            gap=0.002,
            chunk_size=20,
            pieces_per_min=750.0,  # Custom value different from default
            line_eff=0.90,  # Custom value different from default
            dsi_variance=0.05,
            lines=["Line1"],
            cut_strategies=["Strategy1"],
            trim_dollar_value=1.5
        )
        
        # Mock streamlit and load_config
        with patch('portioning.ui.st') as mock_st, \
             patch('portioning.ui.load_config', return_value=mock_config):
            
            # Setup mock streamlit components
            mock_st.sidebar.header = MagicMock()
            mock_st.sidebar.selectbox = MagicMock(return_value="Two-stage (Bulk + Cleanup)")
            mock_st.sidebar.slider = MagicMock(return_value=15)
            mock_st.sidebar.subheader = MagicMock()
            mock_st.sidebar.checkbox = MagicMock(return_value=False)
            mock_st.sidebar.number_input = MagicMock(side_effect=[60, 0.002, 20, 750.0, 0.90])
            
            # Import and call sidebar_controls
            from old.portioning.ui import sidebar_controls
            
            result = sidebar_controls(plants=None, excel_sheets=("Sheet1",))
            
            # Verify that the config values were used
            self.assertEqual(result.pieces_per_min, 750.0)
            self.assertEqual(result.line_eff, 0.90)
            
            # Verify that number_input was called with the config values
            calls = mock_st.sidebar.number_input.call_args_list
            
            # Find the pieces_per_min call (should have value=750.0)
            pieces_per_min_call = None
            line_eff_call = None
            
            for call in calls:
                args, kwargs = call
                if "Pieces per minute" in args:
                    pieces_per_min_call = kwargs
                elif "Line efficiency" in args:
                    line_eff_call = kwargs
            
            self.assertIsNotNone(pieces_per_min_call)
            self.assertIsNotNone(line_eff_call)
            self.assertEqual(pieces_per_min_call['value'], 750.0)
            self.assertEqual(line_eff_call['value'], 0.90)
    
    def test_sidebar_controls_loads_config_at_start(self):
        """Test that sidebar_controls calls load_config at the start of the function."""
        mock_config = AppConfig(
            buckets=[(0, 324)],
            illegal_pairs={"C": ["D"]},
            trim_cap=15,
            time_limit_sec=60,
            gap=0.002,
            chunk_size=20,
            pieces_per_min=600.0,
            line_eff=0.85,
            dsi_variance=0.05,
            lines=["Line1"],
            cut_strategies=["Strategy1"],
            trim_dollar_value=1.5
        )
        
        with patch('portioning.ui.st') as mock_st, \
             patch('portioning.ui.load_config', return_value=mock_config) as mock_load_config:
            
            # Setup minimal mock streamlit components
            mock_st.sidebar.header = MagicMock()
            mock_st.sidebar.selectbox = MagicMock(return_value="Enumeration (interactive)")
            mock_st.sidebar.slider = MagicMock(return_value=15)
            mock_st.sidebar.subheader = MagicMock()
            mock_st.sidebar.radio = MagicMock(side_effect=["Preset buckets", "ALL", "NONE"])
            mock_st.sidebar.number_input = MagicMock(return_value=0)
            
            from old.portioning.ui import sidebar_controls
            
            sidebar_controls(plants=None, excel_sheets=())
            
            # Verify that load_config was called
            mock_load_config.assert_called_once()


if __name__ == '__main__':
    unittest.main()

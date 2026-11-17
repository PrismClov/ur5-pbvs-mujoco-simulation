# -*- coding=utf-8 -*-

# ------------------------------------------------------------------
# File Name:        do_gripper.py
# Description:      DO (Digital Output) gripper control using Dobot API
#                   This implementation uses a configured DO port to control
#                   a simple binary gripper (open/closed).
# ------------------------------------------------------------------

import re
import time
from .dobot_api import DobotApiDashboard


class DOGripperDobot:
    """DO (Digital Output) gripper control class using Dobot API.
    
    This class provides a simple interface for controlling a gripper via
    a digital output port. The gripper position is binary:
    - Position < 500 (below 50%) = closed (DO = 1)
    - Position >= 500 (50% or above) = open (DO = 0)
    
    Attributes:
        dashboard: DobotApiDashboard instance
        do_index: DO port index (1-based, e.g., 1 for DO1)
        current_position: Current gripper position (0 or 1000)
    """
    
    def __init__(self, dashboard: DobotApiDashboard, do_index: int = 1):
        """Initialize DO gripper.
        
        Args:
            dashboard: DobotApiDashboard instance
            do_index: DO port index (1-based, e.g., 1 for DO1, 2 for DO2)
        """
        self.dashboard = dashboard
        self.do_index = do_index
        self.current_position = 0  # Track position (0 = closed, 1000 = open)
        
        # Initialize gripper to open position (DO = 1 for closed)
        self.dashboard.DOInstant(self.do_index, 0)
        print(f"DO gripper initialized on DO{self.do_index}, starting in closed position")
    
    def set_pos(self, val: int, blocking: bool = False):
        """Set the position of the gripper.
        
        Args:
            val: Position value (0-1000)
                 For compatibility, values 0-1000 are supported.
                 Values < 500 (below 50%) map to closed (DO = 1), values >= 500 (50% or above) map to open (DO = 0).
            blocking: If True, wait a short time for DO to settle (default False)
        """
        # Map position to binary state
        # < 500 (below 50%) = closed (DO = 1), >= 500 (50% or above) = open (DO = 0)
        if val < 500:
            target_state = 1  # DO = 1 for closed
            target_position = 0
        else:
            target_state = 0  # DO = 0 for open
            target_position = 1000
        
        # Set DO state
        self.dashboard.DOInstant(self.do_index, target_state)
        self.current_position = target_position
        
        if blocking:
            # Wait a short time for DO to settle (DO changes are immediate, but
            # we wait a bit to ensure the physical gripper has time to respond)
            time.sleep(0.1)
    
    def read_pos(self) -> int:
        """Read the position of the gripper.
        
        Returns:
            Gripper position value (0 = closed, 1000 = open) or None if read fails
        """
        try:
            result = self.dashboard.GetDO(self.do_index)
            # Parse result: format is typically "ErrorID,{value},GetDO(...)"
            # Extract value from result
            do_state = self._parse_do_state(result)
            
            if do_state is None:
                # If we can't read, return current tracked position
                return self.current_position
            
            # Convert DO state to position (DO = 1 means closed = 0, DO = 0 means open = 1000)
            position = 0 if do_state == 1 else 1000
            self.current_position = position
            return position
            
        except Exception as e:
            print(f"Warning: Failed to read DO{self.do_index} state: {e}")
            # Return tracked position as fallback
            return self.current_position
    
    def _parse_do_state(self, result: str) -> int:
        """Parse DO state from GetDO result.
        
        Args:
            result: Result string from GetDO command
            
        Returns:
            DO state (0 or 1) or None if parsing fails
        """
        if not result:
            return None
        
        # Result format: "ErrorID,{value},GetDO(...)"
        # Only extract value from braces format
        match = re.search(r'\{(\d+)\}', result)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                pass
        
        return None
    
    def set_force(self, val: int):
        """Set the force of the gripper (not applicable for DO gripper).
        
        Args:
            val: Force value (ignored for DO gripper)
        """
        # DO gripper doesn't support force control
        print("Warning: DO gripper does not support force control")
        pass
    
    def set_vel(self, val: int):
        """Set the velocity of the gripper (not applicable for DO gripper).
        
        Args:
            val: Velocity value (ignored for DO gripper)
        """
        # DO gripper doesn't support velocity control
        print("Warning: DO gripper does not support velocity control")
        pass
    
    def close(self):
        """Close the gripper connection.
        
        For DO gripper, this sets the DO to closed (1) state.
        """
        try:
            self.dashboard.DOInstant(self.do_index, 1)  # DO = 1 for closed
            self.current_position = 0
            print(f"DO gripper (DO{self.do_index}) set to closed")
        except Exception as e:
            print(f"Warning: Error closing DO gripper: {e}")


if __name__ == "__main__":
    """Example usage of DOGripperDobot."""
    from robot.dobot.dobot_api import DobotApiDashboard
    import time
    
    # Initialize Dobot dashboard
    dashboard = DobotApiDashboard(ip="192.168.5.1", port=29999)
    
    try:
        # Initialize DO gripper on DO1
        gripper = DOGripperDobot(dashboard, do_index=1)
        
        print("Opening gripper (position >= 500)...")
        gripper.set_pos(1000)  # Open gripper (>= 500)
        time.sleep(1.0)
        
        position = gripper.read_pos()
        print(f"Current position: {position}")
        
        print("Closing gripper (position < 500)...")
        gripper.set_pos(0)  # Close gripper (< 500)
        time.sleep(1.0)
        
        print("Testing threshold at 500...")
        gripper.set_pos(499)  # Should close (< 500)
        time.sleep(0.5)
        position = gripper.read_pos()
        print(f"Position 499 -> position: {position}")
        
        gripper.set_pos(500)  # Should open (>= 500)
        time.sleep(0.5)
        position = gripper.read_pos()
        print(f"Position 500 -> position: {position}")
        
        position = gripper.read_pos()
        print(f"Current position: {position}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        try:
            gripper.close()
        except:
            pass

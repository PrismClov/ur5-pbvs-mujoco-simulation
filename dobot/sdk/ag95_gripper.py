# -*- coding=utf-8 -*-

# ------------------------------------------------------------------
# File Name:        ag95_gripper.py
# Description:      AG95 gripper control using Dobot API Modbus RTU
#                   This implementation uses the Dobot robot's RS485
#                   interface to control the AG95 gripper via Modbus RTU,
#                   eliminating the need for direct serial port access.
# ------------------------------------------------------------------

from typing import Optional

from .gripper_base import DobotGripperBase, in_range


class AG95GripperDobot(DobotGripperBase):
    """AG95 gripper control class using Dobot API Modbus RTU.
    
    This class provides the same interface as ag_gripper.py but uses
    the Dobot robot's RS485 interface for Modbus RTU communication
    instead of direct serial port access.
    
    AG95-specific characteristics:
    - Velocity range: 0-1000
    - Position setting: Non-blocking (blocking parameter ignored)
    """
    
    # AG95-specific parameter ranges
    VEL_MIN = 0
    VEL_MAX = 1000
    
    def set_vel(self, val: int):
        """Set the velocity of the gripper.
        
        Args:
            val: Velocity value (0-1000)
        """
        in_range(val, self.VEL_MIN, self.VEL_MAX)
        self._write_register(self.REG_VELOCITY, val)
    
    def set_pos(self, val: int, blocking: bool = False, timeout: Optional[float] = 0.0001):
        """Set the position of the gripper.
        
        Args:
            val: Position value (0-1000)
            blocking: Ignored for AG95 (always non-blocking)
            timeout: Seconds for dashboard reply wait on SetHoldRegs. If not
                None, a failed or missing reply does not raise (fire-and-forget).
        """
        in_range(val, self.POS_MIN, self.POS_MAX)
        self._write_register(self.REG_POSITION, val, timeout=timeout)
        self.gripper_state = val
        # Note: AG95 doesn't support blocking mode, parameter is ignored


if __name__ == "__main__":
    """Example usage of AG95GripperDobot."""
    from .dobot_api import DobotApiDashboard
    import time
    
    # Initialize Dobot dashboard
    dashboard = DobotApiDashboard(ip="192.168.5.1", port=29999)
    
    try:
        # Initialize AG95 gripper via Dobot API
        # Note: If connection fails, try different parity settings:
        # - 'E' (even) - default, standard for Modbus RTU
        # - 'N' (none) - if even parity doesn't work
        # - 'O' (odd) - rarely used
        gripper = AG95GripperDobot(dashboard, slave_id=1, parity='E')
        
        # Use gripper
        print("Setting gripper parameters...")
        gripper.set_force(80)
        gripper.set_vel(500)
        
        print("Opening gripper...")
        gripper.set_pos(1000)  # Open gripper
        time.sleep(1.0)
        
        position = gripper.read_pos()
        print(f"Current position: {position}")
        
        print("Closing gripper...")
        gripper.set_pos(0)  # Close gripper
        time.sleep(1.0)
        
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

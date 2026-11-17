# -*- coding=utf-8 -*-

"""
Base class for Dobot gripper control via Modbus RTU.
Provides common functionality for AG95 and PGE grippers.
"""

import re
import time
from typing import Optional
from abc import ABC, abstractmethod

from .dobot_api import DobotApiDashboard


def in_range(val, val_min, val_max):
    """Check if the value is in the range.
    
    Args:
        val: The value to be checked.
        val_min: The minimum value.
        val_max: The maximum value.
        
    Raises:
        RuntimeError: If the value is out of range.
    """
    if not val_min <= val <= val_max:
        raise RuntimeError('!!! VALUE IS OUT OF RANGE !!!')


class DobotGripperBase(ABC):
    """Base class for Dobot gripper control using Modbus RTU.
    
    This class handles the common Modbus RTU communication via Dobot API
    and provides a framework for gripper-specific implementations.
    
    Attributes:
        dashboard: DobotApiDashboard instance for sending commands
        slave_id: Modbus slave ID (typically 1)
        modbus_index: Index returned by ModbusRTUCreate for subsequent operations
    """
    
    # Register address mappings (common to both AG95 and PGE)
    REG_INIT_STATE = 256      # 0x0100
    REG_FORCE = 257           # 0x0101
    REG_POSITION = 259        # 0x0103
    REG_VELOCITY = 260        # 0x0104
    REG_ABS_ROT = 261         # 0x0105
    REG_ROT_VEL = 263         # 0x0107
    REG_ROT_FORCE = 264       # 0x0108
    REG_REL_ROT = 265         # 0x0109
    REG_FEEDBACK = 512        # 0x0200
    REG_STATE = 513           # 0x0201
    REG_POS_READ = 514        # 0x0202
    
    # Parameter ranges (can be overridden by subclasses)
    FORCE_MIN = 20
    FORCE_MAX = 100
    POS_MIN = 0
    POS_MAX = 1000
    ROT_MIN = -32768
    ROT_MAX = 32767
    ROT_VEL_MIN = 1
    ROT_VEL_MAX = 100
    ROT_FORCE_MIN = 20
    ROT_FORCE_MAX = 100
    
    def __init__(self, dashboard: DobotApiDashboard, slave_id: int = 1, parity: str = 'N'):
        """Initialize gripper using Dobot API.
        
        Args:
            dashboard: DobotApiDashboard instance
            slave_id: Modbus slave ID (typically 1)
            parity: Modbus parity - 'E' (even, default), 'O' (odd), or 'N' (none)
                    Note: Modbus RTU typically uses even parity ('E')
        """
        self.dashboard = dashboard
        self.slave_id = slave_id
        self.modbus_index: Optional[int] = None
        
        # Initialize Modbus connection
        self._setup_modbus(parity)
        
        # Initialize gripper
        self.init_feedback()
    
    def _setup_modbus(self, parity: str):
        """Set up Modbus RTU connection.
        
        Args:
            parity: Modbus parity setting
        """
        # Step 1: Set tool mode to 485 (mode=1 means 485 mode, type is ignored when mode=1)
        # This ensures the tool interface is in RS485 mode, not AI mode
        result = self.dashboard.SetToolMode(mode=1, type=0)
        if self._check_error(result):
            print(f"Warning: SetToolMode returned error (may not be supported on this robot): {result}")
        time.sleep(0.05)
        
        # Step 2: Configure RS485 interface
        # Note: First parameter is baud rate (named 'index' in API but it's actually baud)
        result = self.dashboard.SetTool485(115200, parity=parity, stopbit=1)
        if self._check_error(result):
            raise RuntimeError(f"Failed to configure RS485 interface: {result}")
        print(f"SetTool485 result: {result}")
        time.sleep(0.05)
        
        # Step 3: Enable tool power
        result = self.dashboard.SetToolPower(status=1)
        if self._check_error(result):
            print(f"Warning: SetToolPower returned error (may not be supported): {result}")
        time.sleep(0.2)  # Wait for power to stabilize
        
        # Step 4: Create Modbus RTU connection
        # Close any existing connection first
        try:
            self.dashboard.ModbusClose(0)
            time.sleep(0.1)
        except:
            pass
        
        result = self.dashboard.ModbusCreate(
            ip="127.0.0.1",
            port=60000,
            slave_id=self.slave_id,
            isRTU=1
        )
        print(f"ModbusCreate result: {result}")
        
        if self._check_error(result):
            raise RuntimeError(f"Failed to create Modbus connection: {result}")
        
        self.modbus_index = self._parse_modbus_index(result)
        if self.modbus_index is None:
            raise RuntimeError(f"Failed to parse modbus index from result: {result}")
        
        print(f"Modbus connection established with index: {self.modbus_index}")

        self.gripper_state = self._read_register(self.REG_POS_READ)
        if self.gripper_state != None:
            print(f"Gripper initialized with position: {self.gripper_state / 10}")
        else:
            print(f"Gripper not intialized, reinitalizing")
            self.init_state()
            self.gripper_state = self._read_register(self.REG_POS_READ)
            print(f"Gripper initialized with position: {self.gripper_state / 10}")
    
    def _parse_modbus_index(self, result: str) -> Optional[int]:
        """Parse modbus index from result string.
        
        Args:
            result: Result string from ModbusRTUCreate
            
        Returns:
            Modbus index (typically 0) or None if parsing fails
        """
        if not result:
            return None
        
        # Result format: "ErrorID,{index},ModbusRTUCreate(...)"
        # Try to find number in braces first
        match = re.search(r'\{(\d+)\}', result)
        if match:
            return int(match.group(1))
        
        # If no braces, try to extract number after first comma
        parts = result.split(',')
        if len(parts) > 1:
            try:
                # Second part might be the index
                index_str = parts[1].strip().strip('{}')
                index = int(index_str)
                return index
            except (ValueError, IndexError):
                pass
        
        # If all else fails, try to find any number in the result
        numbers = re.findall(r'\d+', result)
        if numbers and len(numbers) > 1:
            # First number is usually ErrorID, second might be index
            try:
                return int(numbers[1])
            except (ValueError, IndexError):
                pass
        
        # Default to 0 for first connection (most common case)
        print(f"Warning: Could not parse modbus index from '{result}', defaulting to 0")
        return 0
    
    def _parse_register_value(self, result: str) -> Optional[int]:
        """Parse register value from result string.
        
        Args:
            result: Result string from GetHoldRegs
            
        Returns:
            Register value or None if parsing fails
        """
        # Result format: "ErrorID,{value1,value2,...},GetHoldRegs(...)"
        match = re.search(r'\{([^}]+)\}', result)
        if match:
            values = match.group(1).split(',')
            try:
                return int(values[0].strip()) if values else None
            except (ValueError, IndexError):
                pass
        return None
    
    def _check_error(self, result: str) -> bool:
        """Check if result contains an error.
        
        Args:
            result: Result string from Dobot API
            
        Returns:
            True if error occurred, False otherwise
        """
        if not result or len(result.strip()) == 0:
            return True
        
        # Result format: "ErrorID,{...},Command(...)"
        # Try to extract ErrorID from the beginning
        parts = result.split(',')
        if len(parts) > 0:
            try:
                # ErrorID is the first part before comma
                error_id_str = parts[0].strip()
                error_id = int(error_id_str)
                return error_id != 0
            except (ValueError, IndexError):
                # If we can't parse, check if result contains error indicators
                if 'error' in result.lower() or 'fail' in result.lower():
                    return True
                # If we can't determine, assume no error (some commands may return non-standard formats)
                return False
        return False  # If we can't parse, don't assume error
    
    def _write_register(
        self,
        addr: int,
        value: int,
        val_type: str = 'U16',
        timeout: Optional[float] = None,
    ) -> bool:
        """Write a value to a Modbus register.
        
        Args:
            addr: Register address
            value: Value to write
            val_type: Value type ('U16', 'U32', 'F32', 'F64')
            timeout: Optional socket read timeout in seconds for the dashboard
                command (same semantics as DobotApiDashboard.sendRecvMsg).
            
        Returns:
            True if the dashboard reply indicates success, False if the write
            could not be confirmed (including socket timeout when ``timeout`` is set).
            
        Raises:
            RuntimeError: If write fails and ``timeout`` is None (strict mode).
                When ``timeout`` is set, errors are not raised so low-latency
                command streams are not interrupted by missed replies.
        """
        result = self.dashboard.SetHoldRegs(
            self.modbus_index,
            addr=addr,
            count=1,
            valTab=f'{{{value}}}',
            valType=val_type,
            timeout=timeout,
        )
        if self._check_error(result) and timeout is None:
            raise RuntimeError(f"Failed to write register {addr}: {result}")
        return True
    
    def _read_register(self, addr: int, val_type: str = 'U16') -> Optional[int]:
        """Read a value from a Modbus register.
        
        Args:
            addr: Register address
            val_type: Value type ('U16', 'U32', 'F32', 'F64')
            
        Returns:
            Register value or None if read fails
        """
        result = self.dashboard.GetHoldRegs(
            self.modbus_index,
            addr=addr,
            count=1,
            valType=val_type
        )
        if self._check_error(result):
            return None
        return self._parse_register_value(result)
    
    # Common methods (same for both grippers)
    
    def init_state(self):
        """Initialize the gripper.
        
        Writes 165 to register 0x0100 to initialize the gripper state.
        """
        self._write_register(self.REG_INIT_STATE, 165)
    
    def set_force(self, val: int):
        """Set the force of the gripper.
        
        Args:
            val: Force value (20-100)
        """
        in_range(val, self.FORCE_MIN, self.FORCE_MAX)
        self._write_register(self.REG_FORCE, val)
    
    def set_abs_rot(self, val: int):
        """Set the absolute rotation angle of the gripper.
        
        Args:
            val: Rotation angle (-32768 to 32767)
        """
        in_range(val, self.ROT_MIN, self.ROT_MAX)
        self._write_register(self.REG_ABS_ROT, val)
    
    def set_rot_vel(self, val: int):
        """Set the rotation velocity of the gripper.
        
        Args:
            val: Rotation velocity (1-100)
        """
        in_range(val, self.ROT_VEL_MIN, self.ROT_VEL_MAX)
        self._write_register(self.REG_ROT_VEL, val)
    
    def set_rot_force(self, val: int):
        """Set the rotation force of the gripper.
        
        Args:
            val: Rotation force (20-100)
        """
        in_range(val, self.ROT_FORCE_MIN, self.ROT_FORCE_MAX)
        self._write_register(self.REG_ROT_FORCE, val)
    
    def set_rel_rot(self, val: int):
        """Set the relative rotation angle of the gripper.
        
        Args:
            val: Relative rotation angle (-32768 to 32767)
        """
        in_range(val, self.ROT_MIN, self.ROT_MAX)
        self._write_register(self.REG_REL_ROT, val)
    
    def read_state(self) -> Optional[int]:
        """Read the state of the gripper.
        
        Returns:
            Gripper state value or None if read fails
        """
        return self._read_register(self.REG_STATE)
    
    def read_pos(self) -> Optional[int]:
        """Read the position of the gripper.
        
        Returns:
            Gripper position value or None if read fails
        """
        # return self._read_register(self.REG_POS_READ)
        # We return gripper state here to avoid large wait time
        # in _read_register for gripper
        return self.gripper_state
    
    def init_feedback(self):
        """Initialize feedback and wait for ready state.
        
        Reads register 0x0200 to check gripper feedback state.
        Waits until gripper is ready (state != 0 and state != 2).
        
        Raises:
            RuntimeError: If initialization timeout
        """
        max_attempts = 50
        attempt = 0
        
        while attempt < max_attempts:
            state = self._read_register(self.REG_FEEDBACK)
            
            if state is None:
                attempt += 1
                time.sleep(0.1)
                continue
            
            if state == 0:
                # Gripper not initialized, reinitialize
                self.init_state()
                time.sleep(0.1)
            elif state == 2:
                # Gripper busy, wait
                time.sleep(0.1)
            else:
                # Gripper ready
                return
            
            attempt += 1
        
        raise RuntimeError("Gripper feedback initialization timeout")
    
    # Abstract methods - must be implemented by subclasses
    
    @abstractmethod
    def set_vel(self, val: int):
        """Set the velocity of the gripper (range differs by model).
        
        Args:
            val: Velocity value (range depends on gripper model)
        """
        pass
    
    @abstractmethod
    def set_pos(
        self,
        val: int,
        blocking: bool = False,
        timeout: Optional[float] = None,
    ):
        """Set the position of the gripper (blocking behavior differs).
        
        Args:
            val: Position value (0-1000)
            blocking: Whether to wait until position is reached (implementation differs)
            timeout: Optional dashboard command timeout in seconds for the Modbus
                write (AG95); ignored by models that do not use Modbus for position.
        """
        pass
    
    def close(self):
        """Close Modbus connection.
        
        Disconnects from the Modbus slave and releases the master.
        """
        if self.modbus_index is not None:
            try:
                self.dashboard.ModbusClose(self.modbus_index)
            except Exception as e:
                print(f"Warning: Error closing Modbus connection: {e}")
            self.modbus_index = None


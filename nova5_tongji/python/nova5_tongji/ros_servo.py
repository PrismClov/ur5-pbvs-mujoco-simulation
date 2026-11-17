"""Use the existing repository's bounded Nova5 ServoJ orchestration unchanged."""
def main():
    from nova5_visual_servo import interface
    from .tracker import TongjiTargetTracker
    # The original interface explicitly injects its TargetTracker constructor.
    # No other module/class is patched; this affects this process only.
    interface.TargetTracker=TongjiTargetTracker
    interface.main()

if __name__=='__main__': main()

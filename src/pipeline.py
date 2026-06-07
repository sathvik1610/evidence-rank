from phase1 import runner as phase1_runner

def main():
    print("=== EVIDENCE RANK PIPELINE ===")
    phase1_runner.execute()
    # phase2_runner.execute() # To be implemented after review
    print("=== PIPELINE COMPLETE ===")

if __name__ == "__main__":
    main()

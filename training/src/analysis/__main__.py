from analysis.runner import main

if __name__ == "__main__":  # required on Windows: DataLoader workers re-import the main module
    raise SystemExit(main())





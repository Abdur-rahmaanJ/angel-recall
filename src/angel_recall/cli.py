import argparse
from .dashboard.app import app

def main():
    parser = argparse.ArgumentParser(prog="angel-recall")
    subparsers = parser.add_subparsers(dest="command")
    
    memos_parser = subparsers.add_parser("memos", help="Launch the memory dashboard")
    memos_parser.add_argument("--port", type=int, default=5001, help="Port to run the dashboard on")
    memos_parser.add_argument("--debug", action="store_true", help="Run in debug mode")
    
    args = parser.parse_args()
    
    if args.command == "memos":
        print(f"Starting Angel Recall Dashboard on http://127.0.0.1:{args.port}")
        # When running as a package, we need to make sure the app can find templates
        app.run(debug=args.debug, port=args.port)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

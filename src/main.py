import argparse
import sys


class CustomParser(argparse.ArgumentParser):
    def error(self, message):
        if "not allowed with argument" in message:
            message = "Error: Can only run prediction on one file at a time. Please provide a CSV OR txt file."

            self.print_usage(sys.stderr)
            self.exit(2, message)
            return 

        super().error(message)


def parse_args():
    parser = CustomParser(description="A simple CLI to see the percentage of spam emails in a given file.")

    exclusive_group = parser.add_mutually_exclusive_group()
    exclusive_group.add_argument("-f", "--file", help="Enter a file(path) containing an email", type=str)
    exclusive_group.add_argument("-c","--csv", help="Enter a CSV file(path).", type=str)
    args = parser.parse_args()

    if not args.file.endswith(".txt") or args.csv.endswith(".csv"):
        raise TypeError("Unsupported file type detected. Please provide a CSV or txt file.")
    
    return args.file if args.file else args.csv



def main():
    pass


if __name__ == "__main__":
    main()
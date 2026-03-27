import re
import json
import argparse

def extract_success_rates(log_content):
    # Pattern to match success rate lines
    pattern = r'Results for (.+?):\nSuccess rate: (\d\.\d+)'
    
    # Find all matches in the log content
    matches = re.findall(pattern, log_content)
    
    if not matches:
        print("No success rates found in the log content.")
        return None
    
    # Convert to dictionary with task names as keys and success rates as values
    success_rates = {task: float(rate) for task, rate in matches}
    
    # Calculate average success rate
    average_rate = sum(success_rates.values()) / len(success_rates)

    pnpclose_success_rates = {k: v for k, v in success_rates.items() if "Close" in k}
    if len(pnpclose_success_rates) > 0:
        pnpclose_avg_rate = sum(pnpclose_success_rates.values()) / len(pnpclose_success_rates)
    else:
        pnpclose_avg_rate = -9999

    pnponly_success_rates = {k: v for k, v in success_rates.items() if "Close" not in k}
    if len(pnponly_success_rates) > 0:
        pnponly_avg_rate = sum(pnponly_success_rates.values()) / len(pnponly_success_rates)
    else:
        pnponly_avg_rate = -9999
    
    # Prepare the result dictionary
    result = {
        "tasks": success_rates,
        "average_success_rate": {
            "overall": average_rate,
            "PnPClose": pnpclose_avg_rate,
            "PnPOnly": pnponly_avg_rate,
        }
    }
    
    return result

def save_to_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract task success rates from log file and calculate average.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '-i', '--input',
        required=True,
        help='Path to the input log file'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='success_rates.json',
        help='Path to the output JSON file'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print detailed output to console'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Read log file
        with open(args.input, 'r') as file:
            log_content = file.read()
        
        # Extract success rates
        success_data = extract_success_rates(log_content)
        
        if success_data:
            # Save to JSON
            save_to_json(success_data, args.output)
            
            if args.verbose:
                print(f"Success rates extracted and saved to {args.output}")
                print("Results:")
                print(json.dumps(success_data, indent=4))
            else:
                print(f"Success rates extracted and saved to {args.output}")
        else:
            print("No success rates were extracted.")
            
    except FileNotFoundError:
        print(f"Error: Input file '{args.input}' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()
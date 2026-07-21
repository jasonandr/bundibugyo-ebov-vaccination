import json

log_path = "figures/.system_generated/logs/transcript.jsonl"

found = False
with open(log_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'PLANNER_RESPONSE':
            if 'tool_calls' in data:
                for tc in data['tool_calls']:
                    if tc['name'] in ['write_to_file', 'replace_file_content', 'multi_replace_file_content']:
                        args = tc.get('args', {})
                        if 'plot_empirical_data.py' in args.get('TargetFile', ''):
                            if 'CodeContent' in args:
                                print("Found the original plot_empirical_data.py!")
                                with open('plot_empirical_data.py', 'w') as out:
                                    out.write(args['CodeContent'])
                                found = True
                                break
        if found:
            break

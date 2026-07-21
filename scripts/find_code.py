import json

log_path = "figures/.system_generated/logs/transcript.jsonl"

found = False
with open(log_path, 'r') as f:
    for line in f:
        data = json.loads(line)
        if data.get('type') == 'TOOL_CALL':
            # This would be in the tool_calls of a PLANNER_RESPONSE
            pass
        if data.get('type') == 'PLANNER_RESPONSE':
            if 'tool_calls' in data:
                for tc in data['tool_calls']:
                    if tc['name'] in ['write_to_file', 'replace_file_content']:
                        args = tc.get('args', {})
                        if 'estimate_rt.py' in args.get('TargetFile', ''):
                            print("FOUND IT!")
                            if 'CodeContent' in args:
                                print(args['CodeContent'][:500])
                                with open('old_estimate_rt.py', 'w') as out:
                                    out.write(args['CodeContent'])
                            found = True
                            break
        if found:
            break

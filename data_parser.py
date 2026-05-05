import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional


ACTIONS = {
    'cc':  'call',
    'cbr': 'raise',
    'f':   'fold',
    'k':   'check',
    'b':   'bet',
}


def ActionParser(action_str: str) -> Optional[Dict]:

    if not action_str:
        return None

    parts = str(action_str).strip().split()
    if not parts:
        return None

    if parts[0] == 'd':
        if len(parts) >= 4 and parts[1] == 'dh':                                       #hole card parsing
            return {'type': 'deal', 'player': parts[2], 'cards': parts[3]}
      
        if len(parts) >= 3 and parts[1] == 'db':                                  #community card parsing
            return {'type': 'board', 'cards': ' '.join(parts[2:])}
        return None


    if len(parts) < 2:
        return None

    player, actioncode = parts[0], parts[1]


    if actioncode == 'sm':                                         #showdown reveal
        return {'type': 'action', 'player': player,
                'action': 'showdown', 'amount': 0}


    try:
        amount = float(parts[2]) if len(parts) > 2 else 0.0            #normal action 
    except ValueError:
        amount = 0.0

    return {
        'type':   'action',                                  #returns parsed data
        'player': player,
        'action': ACTIONS.get(actioncode, actioncode),
        'amount': amount,
    }


class PHHSParser:

    @staticmethod
    def Parsefile(filepath: str) -> List[Dict]:
        """
        Parse a single .phhs file and return a list of hand dicts.

        Each returned dict has:
          - hand_id:     int section number from the file
          - players:     list of player name strings
          - actions:     list of raw action strings (unparsed)
          - results:     list of integer chip results per player
          - source_file: basename of the originating file
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        hands = []

        sections = re.split(r'\n\[(\d+)\]\n', content)


        for i in range(1, len(sections), 2):
            if i + 1 >= len(sections):
                break

            hand_id = sections[i]                     #section in file
            body = sections[i + 1]


            actionsmatch = re.search(r"actions = \[(.*?)\]", body, re.DOTALL)    #gets axtions    
            if not actionsmatch:
                continue
            actions = re.findall(r"'([^']*)'", actionsmatch.group(1))

            
            playermatch = re.search(r"players = \[(.*?)\]", body)                #gets player names
            players = (re.findall(r"'([^']*)'", playermatch.group(1))
                       if playermatch else [])

            
            Resultmatch = re.search(r"_results = \[(.*?)\]", body)                    #gets results of hand
            results = ([int(x.strip()) for x in Resultmatch.group(1).split(',')]
                       if Resultmatch else [])

            hands.append({
                'hand_id': int(hand_id),
                'players': players,
                'actions': actions,
                'results': results,
                'source_file': os.path.basename(filepath),
            })

        return hands

    @staticmethod
    def Folderparse(folder_path: str, pattern: str = "*.phhs") -> List[Dict]:         #for parsing a folder for valid files

        files = list(Path(folder_path).glob(pattern))
        print(f"Found {len(files)} files matching pattern '{pattern}'")

        allhands = []
        for filepath in files:
            print(f"\nParsing {filepath.name}...", end=" ")
            try:
                hands = PHHSParser.Parsefile(str(filepath))
                allhands.extend(hands)
                print(f"✓ Extracted {len(hands)} hands")
            except Exception as e:
                print(f"✗ Error: {e}")

        print(f"\n{'='*60}")
        print(f"Total hands extracted: {len(allhands)}")
        print(f"{'='*60}")
        return allhands




def ParseStats(hands: List[Dict]) -> Dict:

    if not hands:
        return {}

    actions_per_hand = [len(h['actions']) for h in hands]

    all_players = set()
    action_types: Dict[str, int] = defaultdict(int)
    for h in hands:
        all_players.update(h['players'])
        for a in h['actions']:
            parts = a.split()
            if len(parts) >= 2:
                action_types[parts[1]] += 1

    files = sorted({h['source_file'] for h in hands})

    return {
        'total_hands':          len(hands),
        'total_files':          len(files),
        'files':                files,
        'unique_players':       sorted(all_players),
        'avg_actions_per_hand': sum(actions_per_hand) / len(actions_per_hand),
        'min_actions':          min(actions_per_hand),
        'max_actions':          max(actions_per_hand),
        'action_type_counts':   dict(action_types),
    }


def PrintStats(stats: Dict):
    """Pretty-print the dict returned by get_statistics()."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    print(f"Total hands: {stats['total_hands']}")
    print(f"Total files: {stats['total_files']}")

    files = stats['files']
    head = ', '.join(files[:5])
    tail = f" ... and {len(files) - 5} more" if len(files) > 5 else ""
    print(f"Files: {head}{tail}")

    print(f"\nUnique players: {', '.join(stats['unique_players'])}")
    print(f"\nActions per hand:")
    print(f"  Average: {stats['avg_actions_per_hand']:.1f}")
    print(f"  Min: {stats['min_actions']}")
    print(f"  Max: {stats['max_actions']}")
    print(f"\nAction type distribution:")
    for action_type, count in sorted(stats['action_type_counts'].items(),
                                     key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {action_type}: {count}")

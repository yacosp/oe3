"""read anatta for voe3t"""

import json
oe3_state = json.load(open('var/lib/oe3.json'))
anatta = oe3_state['anatta']
print(anatta['name'] + ' ' + anatta['born_date'][:16])

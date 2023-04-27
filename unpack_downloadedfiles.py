#%%
# import the required modules for unpacking the .gz file
import zlib
import pickle
import compress_pickle
import datetime
import json
# %%
"""
input_struct = [
    {
        '_id': '62fd75d5d2e66101f4c8e48a',
        'meta': {
            'name': 'No name',
            'deviceId': 'TWXY',
            'deviceSerial': '642923300033',
            'deviceDataId': None,
            'chipType': 7,
            'created': '',
            'device': 'TWXY',
        },
        'data': {
            'scan': {
                'C1': {
                    'time': ['2021-01-26T15:00:00.000Z', '2021-01-26T15:00:00.000Z', ...],
                    'rawdata': [...],
                    'fit': [...],
                },
                'C4': {
                    'time': ['2021-01-26T15:00:00.000Z', '2021-01-26T15:00:00.000Z', ...],
                    'rawdata': [...],
                    'fit': [...],
                },
            },
            'temperature': [...],
            'chipInsertion': [...],
            'fluidFill': [...],
        },
        'result': 'Invalid',
        'status': {
            'stat': 'ok',
            'avgTemp': '67.53 C',
            'chipInsertion': '24 / 24',
            'fluid': 'C1-12/12, C4-12/12',
            'channelResult': 'C1-Negative Ct:19.4 Pr:0.1 Sd:0.01,C4-Negative Ct:10.4 Pr:0.4 Sd:0.05',
        },
    },
    {...},
]

output_struct = {'pstraces': {'TWXY': [{'_id': '62fd75d5d2e66101f4c8e48a',
                                        'name': 'No name-C1',
                                        'exp': 'No Exp',
                                        'dtype': 'device-transformed',
                                        'data': {'time': ["<class 'datetime.datetime'>", ...],
                                        'rawdata': [[[1]*120,[1]*120]]*90,
                                        'fit': [{'fx': [0, 1],'fy': [0, 1],'pc': 0.0,'pv': 0.0,'err': 0.0}]*90},
                                        'desc': 'No Desc | {"name": "TWXY", "deviceId": "TWXY", "deviceSerial": "642923300033", "deviceDataId": null, "chipType": 7, "created": "", "device": "TWXY"} | {"stat": "ok", "avgTemp": "67.58 C", "chipInsertion": "29 / 29", "fluid": "C1-14/14, C4-14/14", "channelResult": "C1-Positive Ct:9.4 Pr:1.0 Sd:0.30,C4-Positive Ct:14.4 Pr:1.1 Sd:0.27"} | "Positive"',
                                        '_file': './unspecified_filename_in_load_reader_data',
                                        '_channel': 'TWXY',
                                        'userMarkedAs': 'negative'},
                                        { '_id': '62fd75d5d2e66101f4c8e48a',
                                        'name': 'No name-C4',
                                        'exp': 'No Exp',
                                        'dtype': 'device-transformed',
                                        'data': {'time': ["<class 'datetime.datetime'>", ...],
                                        'rawdata': [[[1]*120,[1]*120]]*90,
                                        'fit': [{'fx': [0, 1],'fy': [0, 1],'pc': 0.0,'pv': 0.0,'err': 0.0}]*90},
                                        'desc': 'No Desc | {"name": "TWXY", "deviceId": "TWXY", "deviceSerial": "642923300033", "deviceDataId": null, "chipType": 7, "created": "", "device": "TWXY"} | {"stat": "ok", "avgTemp": "67.58 C", "chipInsertion": "29 / 29", "fluid": "C1-14/14, C4-14/14", "channelResult": "C1-Positive Ct:9.4 Pr:1.0 Sd:0.30,C4-Positive Ct:14.4 Pr:1.1 Sd:0.27"} | "Positive"',
                                        '_file': './unspecified_filename_in_load_reader_data',
                                        '_channel': 'TWXY',
                                        'userMarkedAs': 'negative'}],
                             'isReaderData': 0}
"""
#%%
def reltime_to_datetime(reltime: float) -> datetime.datetime:
    """
    convert relative time to datetime
    """
    today = datetime.datetime.today().strftime('%Y-%m-%d').split('-')
    datetime_from_timestamp = datetime.datetime.fromtimestamp(reltime * 60)
    return datetime_from_timestamp.replace(year=int(today[0]),month=int(today[1]),day=int(today[2]))

#define a function that takes input_struct and returns output_struct
def restructure_data(data):
    """
    take the .gz file and restructure it into picklez files that can be read by the trainer
    """
    readers = set()
    for datum in data:
        readers.add(datum['meta']['device'])
    output_struct = {'pstraces': {reader:[] for reader in readers}, 'isReaderData': 1}

    for datum in data:
        if not datum['data']:
            continue
        reader = datum['meta']['device']
        for channel in datum['data']['scan']:
            output_struct['pstraces'][reader].append({'_id': datum['_id'],
                                                    'name': datum['meta']['name'] + '-' + channel,
                                                    'exp': 'No Exp',
                                                    'dtype': 'device-transformed',
                                                    'data': {'time': [reltime_to_datetime(reltime) for reltime in datum['data']['scan'][channel]['time']],
                                                            'rawdata': datum['data']['scan'][channel]['rawdata'],
                                                            'fit': datum['data']['scan'][channel]['fit']},
                                                    'desc': 'No Desc | ' + json.dumps(datum['meta']) + ' | ' + json.dumps(datum['status']) + ' | ' + json.dumps(datum['result']),
                                                    '_file': './unspecified_filename_in_load_reader_data',
                                                    '_channel': reader,
                                                    'userMarkedAs': None})
    return output_struct

def gz_to_picklez(file):
    """
    take a .gz file and unpack it into a .picklez file
    """
    # read the .gz file
    with open(file, 'rb') as f_in:
        data = f_in.read()
    dec = zlib.decompress(data)
    data = pickle.loads(dec)
    # restructure the data
    data_out = restructure_data(data)
    # save the unpacked file to a .picklez using pickle
    with open(file.replace('.gz', '.picklez'), 'wb') as f_out:
        compress_pickle.dump(data_out, f_out, compression='gzip')

#%%
if __name__ == '__main__':
    while True:            
        file = input('Enter the of file to be converted: ').strip('&').strip().strip('"')
        if file == 'quit':
            break
        gz_to_picklez(file)
        print('Done! Starting the next file...(enter "quit" to exit the program)\n')
import argparse
from arena_server.wifi_environment import WiFiEnvironment
from multiprocessing import Process


def run_port(args, port):    
    while True:
        try:
            env = WiFiEnvironment(args.config_file, args.evaluation_mode, args.evaluation_time, args.send_to_api, port=port)
            env.run()
        except:
            print('env reset')        


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('T', 'True', 't', 'true'):
            return True
        elif v.lower() in ('F', 'False', 'f', 'false'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser.add_argument('--config_file', '-c', type=str, default='phase_1.yaml')
    parser.add_argument('--evaluation_mode', '-m', type=str2bool, default=False)
    parser.add_argument('--evaluation_time', '-t', type=int, default=10000000)
    parser.add_argument('--send_to_api', '-s', type=str2bool, default=False)
    args = parser.parse_args()
    print(args.config_file, args.evaluation_mode, args.evaluation_time, args.send_to_api)

    processes = []
    for i in range(7500, 7510):
        p = Process(target=run_port, args=(args, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

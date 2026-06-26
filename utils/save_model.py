import os
import torch

def save_model(saved_name, net, logger, overwrite=False, saved_root='./saved_models'):
    saved_name = saved_name + '.pt'

    if not os.path.exists(saved_root):
        os.makedirs(saved_root)

    saved_path = os.path.join(saved_root, saved_name)

    if not os.path.exists(saved_path):
        torch.save(net.state_dict(), saved_path)
        if logger is not None:
            logger.info(f'Model is saved at {saved_path}', pos="blank")
        else:
            print(f'Model is saved at {saved_path}')
    else:
        if overwrite:
            torch.save(net.state_dict(), saved_path)
            if logger is not None:
                logger.info(f'File is overwrote and Model is saved at {saved_path}', pos="blank")
            else:
                print(f'Model is overwritten at {saved_path}')
        else:
            if logger is not None:
                logger.info(f'Model exists at {saved_path} and cannot be overwritten because of "self.overwrite" is {overwrite}', pos="blank")
            else:
                print(
                f'Model exists at {saved_path} and cannot be overwritten because of "self.overwrite" is {overwrite}')
            while True:
                answer = input('Do you want to overwrite it? y/n')
                if answer.upper() == 'Y':
                    torch.save(net.state_dict(), saved_path)
                    break
                elif answer.upper() == 'N':
                    answer2 = input('Please rename the file, or input nothing to quit:')
                    if answer2.upper() != "" :
                        saved_name = answer2
                        save_model(saved_root, saved_name, net, logger, overwrite)
                        return
                    else:
                        if logger is not None:
                            logger.info('Model is not overwritten and saved and the program is over!', pos="blank")
                        else:
                            print('Model is not overwritten and saved and the program is over!')
                        return
                else:
                    print(f'input must be "y" or "n" but got {answer}')

    return
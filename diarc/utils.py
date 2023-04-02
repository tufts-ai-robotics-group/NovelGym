import datetime
import socket
import time
import traceback
import os

last_data: bytes = b""
should_recover = os.environ.get("DEBUG") != "true"

if not should_recover:
    print("Debug mode enabled.")

class bcolors:
    # https://stackoverflow.com/questions/287871
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def recv_socket_data(sock):
    global last_data
    msg = b""
    done = False
    while not done:
        try:
            slice_msg = sock.recv(4096, socket.MSG_PEEK)
            if b'\n' in slice_msg:
                index = slice_msg.find(b'\n')
                msg += sock.recv(index)
                sock.recv(1)
                done = True
            else:
                msg += sock.recv(4096)
        except BlockingIOError as e:
            time.sleep(0.05)
    last_data = msg
    return msg


def has_substr_in_buffer_blocking(sock: socket.socket, substr: str, ignore_case=False, num_bytes=4096):
    if ignore_case:
        substr = substr.lower()
    while True:
        try:
            slice_msg = sock.recv(num_bytes, socket.MSG_PEEK)
            if slice_msg != b"":
                if ignore_case:
                    slice_msg_decoded = slice_msg.decode("unicode_escape").lower()
                else:
                    slice_msg_decoded = slice_msg.decode("unicode_escape")
                if substr in slice_msg_decoded:
                    return True
                else:
                    return False
        except BlockingIOError:
            pass
        except Exception as e:
            raise e


def has_substr_in_buffer(sock: socket.socket, substr: str, ignore_case=False):
    if ignore_case:
        substr = substr.lower()
    try:
        slice_msg = sock.recv(4096, socket.MSG_PEEK)
        if ignore_case:
            slice_msg_decoded = slice_msg.decode("unicode_escape").lower()
        else:
            slice_msg_decoded = slice_msg.decode("unicode_escape")
        if substr in slice_msg_decoded:
            return True
        else:
            return False
    except BlockingIOError:
        return False
    except Exception as e:
        raise e


def socket_buf_is_not_empty(sock: socket.socket):
    """
    Tries to get data from the socket,
    if it's not empty, return True plus the content.
    if it's empty, return False plus empty bytes.
    """
    msg = b""

    # set socket to be unblocking so that we will immediately return
    # if there's nothing in the socket buffer.
    
    if has_substr_in_buffer(sock, "replan", ignore_case=True):
        msg = recv_socket_data(sock)
        has_content = True
    else:
        has_content = False

    # set the socket back to blocking so that in the future we will still
    # receive the packet as intended.
    return (has_content and msg != b""), msg


def save_pddl(pddl_content: str, task_name: str):
    """
    Saves PDDL content for debugging of diarc.
    """
    try:
        curr_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pddl_dump_{task_name}_{curr_time}.pddl"
        with open(filename, 'w') as f:
            f.write(pddl_content)
    except IOError:
        pass


def save_failed_json(json_content: str):
    try:
        curr_time = str(round(time.time()))
        filename = f"error_dump_{curr_time}.json"
        with open(f"error_dump_{curr_time}.json", 'w') as f:
            f.write(json_content)
        return filename
    except IOError:
        pass

def error_recovery(func):
    if os.environ.get("DEBUG") == "true":
        errors = (ValueError, KeyError)
    else:
        errors = ()

    def wrapped_func(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except errors as e:
            print(e)
            if not should_recover:
                raise e from e
            print(bcolors.WARNING + "Encountered error: ")
            print(e)
            print(traceback.format_exc() + bcolors.ENDC)
            filename = save_failed_json(last_data.decode("unicode_escape"))
            print(f"saved json to {filename}.")
            print()
    return wrapped_func

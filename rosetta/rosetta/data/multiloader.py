#
# Copyright (c) 2017-2023 NVIDIA CORPORATION. All rights reserved.
# This file is part of the WebDataset library.
# See the LICENSE file for licensing terms (BSD-style).
#

"""An alternative to DataLoader using ZMQ.
This implements MultiLoader, an alternative to DataLoader when torch
is not available. Subprocesses communicate with the loader through
ZMQ, provided for high performance multithreaded queueing.
"""

import multiprocessing as mp
import pickle
import uuid
import weakref
import threading
import queue
import logging

import zmq
import os
from multiprocessing import Lock

the_protocol = pickle.HIGHEST_PROTOCOL

all_pids = weakref.WeakSet()


class EOF:
    """A class that indicates that a data stream is finished."""

    def __init__(self, **kw):
        """Initialize the class with the kw as instance variables."""
        self.__dict__.update(kw)

class BufferState():
    def __init__(self, max_size):
        self.q = mp.Queue(maxsize=max_size) 

    def increment(self):
        self.q.put(0)

    def decrement(self):
        self.q.get()

    def get_len(self):
        return self.q.qsize()

    def reset(self):
        while not self.q.empty():
            self.q.get_nowait()

def async_depickler(out_queue, in_zmq_pipe, stop_signal):
    while True:
        if stop_signal:
            return
        data = in_zmq_pipe.recv()
        data = pickle.loads(data)
        out_queue.put(data)

def reader(dataset, sockname, index, num_workers, buf_state, signal_state):
    """Read samples from the dataset and send them over the socket.
    :param dataset: source dataset
    :param sockname: name for the socket to send data to
    :param index: index for this reader, using to indicate EOF
    """
    global the_protocol
    os.environ["WORKER"] = str(index)
    os.environ["NUM_WORKERS"] = str(num_workers)
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.PUSH)
    # sock.set_hwm(prefetch_buffer_len)
    sock.connect(sockname)
    for sample in dataset:
        buf_state.increment()
        data = pickle.dumps(sample, protocol=the_protocol)
        sock.send(data)
        if signal_state.value != 0:
            break
    sock.send(pickle.dumps(EOF(index=index)))
    sock.close()


class MultiLoader:
    """Alternative to PyTorch DataLoader based on ZMQ."""

    def __init__(
        self, dataset, workers=4, verbose=True, nokill=True, prefix="/tmp/_multi-", prefetch_buf_max=128
    ):
        """Create a MultiLoader for a dataset.
        This creates ZMQ sockets, spawns `workers` subprocesses, and has them send data
        to the socket.
        :param dataset: source dataset
        :param workers: number of workers
        :param verbose: report progress verbosely
        :param nokill: don't kill old processes when restarting (allows multiple loaders)
        :param prefix: directory prefix for the ZMQ socket
        """
        self.dataset = dataset
        self.workers = workers
        self.orig_workers = workers
        self.max_workers = workers * 2
        self.retune_period = 100
        self.verbose = verbose
        self.pids = []
        self.socket = None
        self.ctx = zmq.Context.instance()
        self.nokill = nokill
        self.prefix = prefix
        # self.prefetch_buf_per_worker = prefetch_buf_per_worker
        self.prefetch_buf_max = prefetch_buf_max
        self.buf_state = BufferState(prefetch_buf_max)
        self.signal_vals = []
        self.buffer_low_mark=int(prefetch_buf_max * .15)
        assert self.buffer_low_mark < self.prefetch_buf_max
        self.depickled_queue = queue.Queue()
        self.async_depickler = None
        self.async_depickler_stop_signal = False
        self.has_started = False

    def kill(self):
        """kill."""
        self.async_depickler_stop_signal = True
        self.async_depickler = None

        for pid in self.pids:
            if pid is None:
                continue
            print("killing", pid)
            pid.kill()

        for pid in self.pids:
            # pid.join(1.0)
            if pid is None:
                continue
            print("joining", pid)
            pid.join()

        self.pids = []
        if self.socket is not None:
            print("closing", self.socket)
            self.socket.close(linger=0)
            print("Closed")
        self.socket = None
        self.buf_state.reset()

    def __iter__(self):
        """Return an iterator over this dataloader."""
        if self.has_started: 
            logging.warning("RESTARTING LOADER")
        if not self.nokill:
            self.kill()
        if not self.has_started or not self.nokill:
            self.sockname = "ipc://" + self.prefix + str(uuid.uuid4())
            self.socket = self.ctx.socket(zmq.PULL)
            self.socket.set(zmq.LINGER, 0)
            self.socket.bind(self.sockname)
            if self.verbose:
                print("#", self.sockname)
            self.pids = [None] * self.max_workers
            self.signal_vals = [None] * self.max_workers
            for index in range(self.workers):
                signal = mp.Value('i', 0)
                args = (self.dataset, self.sockname, index, self.workers, self.buf_state, signal)
                self.pids[index] = mp.Process(target=reader, args=args)
                self.signal_vals[index] = signal
            all_pids.update(self.pids[:self.workers])
            for pid in self.pids:
                if pid is not None:
                    pid.start()

            # Async depickler setup
            self.async_depickler_stop_signal = False
            self.async_depickler = threading.Thread(target=async_depickler, args=(self.depickled_queue, self.socket, self.async_depickler_stop_signal), daemon=True)
            self.async_depickler.start()

        self.has_started = True
        count = 0
        while self.pids.count(None) < len(self.pids):
            sample = self.depickled_queue.get(block=True)
            if isinstance(sample, EOF):
                if self.verbose:
                    print("# subprocess finished", sample.index)
                self.pids[sample.index].join(1.0)
                self.pids[sample.index] = None
            else:
                self.buf_state.decrement()
                yield sample
            count += 1

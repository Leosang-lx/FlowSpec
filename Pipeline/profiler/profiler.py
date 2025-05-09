import torch
import time
import functools
import pickle
import gc
from contextlib import contextmanager

sep = '\n' + '-' * 50 + '\n'
star = '\n' + '*' * 50 + '\n'

def is_strictly_ascending(arr):
    return all(arr[i] < arr[i+1] for i in range(len(arr)-1))

def save_as(obj, fname):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

class Profiler:
    def __init__(self):
        self.time_events = {}
        self.mem_events = {}
        # [update] for cumulative timing
        self.test_cnt = 0
        self.cumulative_time_events = []
        self.temp_timestamps = None
        self.temp_events = None

    def time_start(self, name, stream=None, cpu=False):
        """
        Start recording time for a named section. Supports multiple starts.
        """

        self.start_tag = 'pipeline' in name
        if self.start_tag:
            self.cumulative_time_events.append(
                {
                    'timestamp': [],
                    'events': []
                }
            )
            self.temp_timestamps = self.cumulative_time_events[-1]['timestamp']
            self.temp_events = self.cumulative_time_events[-1]['events']
            self.test_cnt += 1
        else:
            assert len(self.cumulative_time_events) == self.test_cnt, "cumulative_time_events is not empty"
        if name not in self.time_events:
            self.time_events[name] = {'start': [], 'end': [], 'elapsed': 0.0, 'cpu': cpu}
        assert len(self.time_events[name]['start']) == len(self.time_events[name]['end']), \
            f"Cannot start '{name}' as there are more starts than stops"
        
        # [update] 同步cpu时间
        if not cpu:
            torch.cuda.synchronize()
        start_event = time.perf_counter()

        # if cpu:
        #     start_event = time.perf_counter()
        # else:
        #     start_event = torch.cuda.Event(enable_timing=True)
        #     if stream is not None:
        #         start_event.record(stream)
        #     else:
        #         start_event.record()
        
        # [update] for cumulative timing
        self.temp_timestamps.append(start_event)
        # only append the name when enter the event
        self.temp_events.append(f'{name}')
        self.time_events[name]['start'].append(start_event)

    def time_stop(self, name, stream=None, cpu=False):
        """
        Stop recording time for a named section. Accumulates total time for multiple stops.
        """
        assert name in self.time_events, f"No events recorded for '{name}'"
        assert len(self.time_events[name]['start']) - 1 == len(self.time_events[name]['end']), \
            f"Cannot stop '{name}' as there are more stops than starts"

        # [update] 同步cpu时间
        if not cpu:
            torch.cuda.synchronize()
        end_event = time.perf_counter()

        # if cpu:
        #     end_event = time.perf_counter()
        # else:
        #     end_event = torch.cuda.Event(enable_timing=True)
        #     if stream is not None:
        #         end_event.record(stream)
        #     else:
        #         end_event.record()
        
        # [update] for cumulative timing
        self.temp_timestamps.append(end_event)
        self.time_events[name]['end'].append(end_event)
        
    def elapsed_time(self, name):
        """
        Get the total accumulated time for a specific named section.
        Syncs and stores the result, clearing start and end events after.
        """
        if name not in self.time_events:
            raise ValueError(f"No events recorded for '{name}'")
        
        # Check if CPU timing or CUDA event timing is used
        cpu = self.time_events[name]['cpu']
        total_time = self.time_events[name]['elapsed']
        total_count = len(self.time_events[name]['start'])
        # Accumulate new times
        # if cpu:
        for start, end in zip(self.time_events[name]['start'], self.time_events[name]['end']):
            total_time += (end - start) * 1000  # Convert seconds to ms
        # else:
        #     torch.cuda.synchronize()
        #     for start, end in zip(self.time_events[name]['start'], self.time_events[name]['end']):
        #         total_time += start.elapsed_time(end)

        # Store the accumulated time and clear the events
        self.time_events[name]['elapsed'] = total_time
        self.time_events[name]['start'] = []
        self.time_events[name]['end'] = []
        avg_time = total_time / total_count if total_count > 0 else 0
        return total_time, avg_time
    
    def delete_time_events(self, name):
        """
        Delete the time events for a named section.
        """
        if name in self.time_events:
            for start, end in zip(self.time_events[name]['start'], self.time_events[name]['end']):
                del start, end
            del self.time_events[name]
            gc.collect()
            torch.cuda.empty_cache()
            
    
    def memory_start(self, name, device='cuda:1', cpu=False):
        """
        Start recording memory for a named section. Supports multiple starts.
        """
        if name not in self.mem_events:
            self.mem_events[name] = {'start': [], 'end': [], 'additional': (0, 0, 0, 0), 'cpu': cpu}
        assert len(self.mem_events[name]['start']) == len(self.mem_events[name]['end']), \
            f"Cannot start '{name}' as there are more starts than stops"
        
        if cpu:
            raise ValueError("CPU memory profiling is not supported")
        else:
            torch.cuda.reset_peak_memory_stats(device=device) 
            allocated = torch.cuda.memory_allocated(device=device) / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device=device) / (1024 * 1024)
            max_reserved = torch.cuda.max_memory_reserved(device=device) / (1024 * 1024)
            self.mem_events[name]['start'].append((allocated,  max_allocated, reserved, max_reserved))

    def memory_stop(self, name, device='cuda:1', cpu=False):
        """
        Stop recording memory for a named section. Accumulates total memory for multiple stops.
        """
        if name not in self.mem_events:
            raise ValueError(f"No events recorded for '{name}'")
        assert len(self.mem_events[name]['start']) - 1 == len(self.mem_events[name]['end']), \
            f"Cannot stop '{name}' as there are more stops than starts"
        
        if cpu:
            raise ValueError("CPU memory profiling is not supported")
        else:
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated(device=device) / (1024 * 1024)
            max_allocated = torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)
            reserved = torch.cuda.memory_reserved(device=device) / (1024 * 1024)
            max_reserved = torch.cuda.max_memory_reserved(device=device) / (1024 * 1024)
            self.mem_events[name]['end'].append((allocated, max_allocated, reserved, max_reserved))
    
    def additional_memory(self, name):
        """
        Get the additional memory allocated for a named section.
        """
        if name not in self.mem_events:
            raise ValueError(f"No events recorded for '{name}'")
        
        cpu = self.mem_events[name]['cpu']
        total_additional = (0, 0, 0, 0)
        total_count = len(self.mem_events[name]['start'])
        if cpu:
            raise ValueError("CPU memory profiling is not supported")
        else:
            for start, end in zip(self.mem_events[name]['start'], self.mem_events[name]['end']):
                total_additional = (total_additional[0] + (end[0] - start[0]), 
                                    total_additional[1] + (end[1] - start[1]), 
                                    total_additional[2] + (end[2] - start[2]), 
                                    total_additional[3] + (end[3] - start[3]))
        avg_additional = tuple(additional / total_count for additional in total_additional)
        return total_additional, avg_additional
            
    @contextmanager
    def time_context(self, name, stream=None, cpu=False):
        try:
            self.time_start(name, stream, cpu)
            yield
        finally:
            self.time_stop(name, stream, cpu)
    
    @contextmanager
    def memory_context(self, name, device='cuda:1', cpu=False):
        try:
            self.memory_start(name, device, cpu)
            yield
        finally:
            self.memory_stop(name, device, cpu)
        
    @contextmanager
    def profile_context(self, name, stream=None, device='cuda:1', cpu=False):
        try:
            # self.memory_start(name, device, cpu)
            self.time_start(name, stream, cpu)
            yield
        finally:
            self.time_stop(name, stream, cpu)
            # self.memory_stop(name, device, cpu)

    def get_all_elapsed_times(self):
        """
        Returns a dictionary of the accumulated elapsed times for each recorded event.
        """
        total_times = {}
        avg_times = {}
        for name in self.time_events:
            total_times[name], avg_times[name] = self.elapsed_time(name)
        return total_times, avg_times
    
    def get_all_additional_memory(self):
        """
        Returns a dictionary of the accumulated additional memory for each recorded event.
        """
        total_additional = {}
        avg_additional = {}
        for name in self.mem_events:
            total_additional[name], avg_additional[name] = self.additional_memory(name)
        return total_additional, avg_additional

    def reset(self):
        """
        Reset all stored events.
        """
        self.time_events = {}
        self.mem_events = {}
        
    def print_all_elapsed_times(self):
        """
        Print all recorded elapsed times.
        """
        total_time, avg_time = self.get_all_elapsed_times()
        for name, time in total_time.items():
            print(star + f"{name}" + star + f"total time: {time:.2f} ms, avg time: {avg_time[name]:.2f} ms")

    def print_all_additional_memory(self):
        """
        Print all recorded additional memory.
        """
        total_additional, avg_additional = self.get_all_additional_memory()
        for name, additional in total_additional.items():
            total_additional_str = f" allocated: {additional[0]:.2f} MB \n max allocated: {additional[1]:.2f} MB \n reserved: {additional[2]:.2f} MB \n max reserved: {additional[3]:.2f} MB"
            avg_additional_str = f" avg allocated: {avg_additional[name][0]:.2f} MB \n avg max allocated: {avg_additional[name][1]:.2f} MB \n avg reserved: {avg_additional[name][2]:.2f} MB \n avg max reserved: {avg_additional[name][3]:.2f} MB"
            
            print(star + f"{name}" + star + f"{total_additional_str}" + sep + f"{avg_additional_str}")

    def print_all_events(self):
        """
        Print all recorded events.
        """
        self.print_all_elapsed_times()
        # self.print_all_additional_memory()

prof = Profiler()
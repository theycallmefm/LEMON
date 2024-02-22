from argparse import ArgumentParser
import numpy as np
import random

class EditViewSampler:
    modes = ['random', 'sequential', 'sequential_shuffled']

    def __init__(self, views, mode, views_per_iter):
        if not mode in self.modes:
            raise ValueError(f"Unknown mode '{mode}'. Available modes are {', '.join(self.modes)}")
        self.mode = mode
        self.views_per_iter = views_per_iter
        self.num_views = len(views)

        self.current_index = 0
        if self.mode == "sequential":
            self.index_buffer = list(range(self.num_views))
        elif self.mode == "random":
            self.index_buffer = random.sample(range(self.num_views), self.num_views)

    @staticmethod
    def add_arguments(parser: ArgumentParser):
        group = parser.add_argument_group("View Sampling")
        group.add_argument('--view_sampling_mode', type=str, choices=EditViewSampler.modes, default='sequential', help="Mode used to sample views.")
        group.add_argument('--views_per_iter', type=int, default=1, help="Number of views used per iteration.")
        
    @staticmethod
    def get_parameters(args):
        return { 
            "mode": args.view_sampling_mode,
            "views_per_iter": args.views_per_iter
        }
    
    # def __call__(self, views, new_views):
    #     sampled_views, edited_views, selected_indices = [], [], []
    #     for i in range(self.views_per_iter):
    #         sampled_views += [views[self.index_buffer[self.current_index]]]
    #         edited_views += [new_views[self.index_buffer[self.current_index]]]
    #         selected_indices += [self.index_buffer[self.current_index]]
    #         self.current_index+=1
    #         if self.current_index == self.num_views:
    #             if self.mode == "sequential":
    #                 self.index_buffer = list(range(self.num_views))
    #             elif self.mode == "random":
    #                 self.index_buffer = random.sample(range(self.num_views), self.num_views)
    #             self.current_index = 0
    #     return sampled_views, edited_views, selected_indices

    def __call__(self, views, new_views):
        if self.mode == 'random':
            selected_indices = np.random.choice(self.index_buffer, self.views_per_iter, replace=False)
            sampled_views, edited_views = [], []
            for i in range(self.views_per_iter):
                sampled_views += [views[selected_indices[i]]]
                edited_views += [new_views[selected_indices[i]]]
            return sampled_views, edited_views, selected_indices
        elif self.mode == 'sequential':
            selected_indices = []
            sampled_views, edited_views = [], []
            for _ in range(self.views_per_iter):
                sampled_views += [views[self.current_index]]
                edited_views += [new_views[self.current_index]]
                selected_indices += [self.current_index]
                self.current_index = (self.current_index + 1) % self.num_views
            return sampled_views, edited_views, selected_indices
        elif self.mode == 'sequential_shuffled':
            # Select N views by traversing the full set of views sequentially, but in random order
            # (Each time the full set of views is traversed, randomly shuffle to create a new order)
            selected_indices = []
            sampled_views, edited_views = [], []
            for _ in range(self.views_per_iter):
                view_index = self.index_buffer[self.current_index]
                sampled_views += [views[view_index]]
                edited_views += [new_views[view_index]]
                selected_indices += [view_index]
                self.current_index = (self.current_index + 1) 
                if self.current_index >= self.num_views:
                    random.shuffle(self.index_buffer)
                    self.current_index = 0
            
            return sampled_views, edited_views, selected_indices
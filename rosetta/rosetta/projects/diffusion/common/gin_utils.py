# Copyright (c) 2022-2023 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools


def call(func, args=None, **keywords):
    '''
    Same as partial, except will call the output of partial assuming all args are already specified
    '''
    return partial(func=func, args=args, **keywords)()


def partial(func, args=None, **keywords):
    '''
    Use this in gin-configs if you want to use functools.partial.
    
    In gin-config, there is no support for positional arguments as bindings. Also, to use functools.partial,
    you must pass the function as a positional argument, so this function exists as a workaround. To be
    concrete, none of the following would work in a gin file
    
    ```
    # Case 1
    train/functools.partial:
      my_awesome_fn
      a_kwarg=3
      
    # Case 2
    train/functools.partial:
      func=my_awesome_fn
      a_kwarg=3

    # Case 3
    train/functools.partial:
      func=my_awesome_fn
      {'a_dict': 'as a pos_arg'}
    ```
    
    This function can support Case 2, and Case 1 needs to be written like Case 2 to work. As for
    Case 3, you'd need to write it:
    ```
    train/functools.partial:
      func=my_awesome_fn
      args=[{'a_dict': 'as a pos_arg'}]
    
    The list is required so that we can unpack it as positional args to functools.partial
    ```
    '''
    if args is not None and not isinstance(args, (list, tuple)):
      raise TypeError(f'If you specify args, it must be a tuple or list, but you passed {type(args)}: {args}')
    
    if args is None:
      args = []
    return functools.partial(func, *args, **keywords)

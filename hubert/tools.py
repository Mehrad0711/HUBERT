#!/usr/bin/env python3
# Copyright 2019
#
# Author: Mehrad Moradshahi <mehrad@cs.stanford.edu>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#  list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#  this list of conditions and the following disclaimer in the documentation
#  and/or other materials provided with the distribution.
#
# * Neither the name of the copyright holder nor the names of its
#  contributors may be used to endorse or promote products derived from
#  this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import sys
from . import run_model

subcommands = {
    'run_model': ('Train/ Evaluate/ Test a model', run_model.run_main),
    # 'convert-to-logical-froms': ('Convert to logical forms (for SQL tasks)', convert_to_logical_forms.main),
    # 'train': ('Train a model', train.main),
    # 'predict': ('Evaluate a model, or compute predictions on a test dataset', predict.main),
    # 'server': ('Export RPC interface to predict', server.main),
    # 'cache-embeddings': ('Download and cache embeddings', cache_embeddings.main)
}

def usage():
    print('Usage: %s SUBCOMMAND [OPTIONS]' % (sys.argv[0]), file=sys.stderr)
    print(file=sys.stderr)
    print('Available subcommands:', file=sys.stderr)
    for subcommand, (help_text, _) in subcommands.items():
        print('  %s - %s' % (subcommand, help_text), file=sys.stderr)
    sys.exit(1)

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in subcommands:
        usage()
        return

    main_fn = subcommands[sys.argv[1]][1]
    canned_argv = ['hubert-' + sys.argv[1]] + sys.argv[2:]
    main_fn(canned_argv)

if __name__ == '__main__':
    main()
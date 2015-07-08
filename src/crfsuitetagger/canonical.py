# This file is part of CRFSuiteTagger.
#
# CRFSuiteTagger is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CRFSuiteTagger is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CRFSuiteTagger.  If not, see <http://www.gnu.org/licenses/>.
__author__ = 'Aleksandar Savkov'

# Generally, this doesn't seem like a very good idea, unless you really need to
# normalise your data. Usually the performance is either the same or lower.

# How does this work?
# Add a regexp as the key and a replacement string as the value, and they will
# be replaced in during the feature extraction. Note that this will not affect
# the output or the input in any way. If left empty nothing is replaced.

REPLACEMENTS = {
    # r'~+':      '<redacted>',
    # r'\+\++':   '++',
    # r'\?\?+':   '?',
    # r'\.\.\.+': '...',
    # r'\d+':     '<number>'
}
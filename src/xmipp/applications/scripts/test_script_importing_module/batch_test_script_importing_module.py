#!/usr/bin/env python3
# ***************************************************************************
# *
# * Authors:     David Maluenda (dmaluenda@cnb.csic.es)
# *
# *
# * This program is free software; you can redistribute it and/or modify
# * it under the terms of the GNU General Public License as published by
# * the Free Software Foundation; either version 2 of the License, or
# * (at your option) any later version.
# *
# * This program is distributed in the hope that it will be useful,
# * but WITHOUT ANY WARRANTY; without even the implied warranty of
# * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# * GNU General Public License for more details.
# *
# * You should have received a copy of the GNU General Public License
# * along with this program; if not, write to the Free Software
# * Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA
# * 02111-1307  USA
# *
# *  All comments concerning this program package may be sent to the
# *  e-mail address 'scipion@cnb.csic.es'
# ****************************************************************************
""" This is a test and an example of a script which is importing
    from py_xmipp library.
"""

import sys


def green(txt):
    return "\033[92m"+txt+"\033[0m"

def red(text):
    return "\033[91m"+text+"\033[0m"

print(green("[ RUN      ]")+" test_script_importing_module")

try:
    print("Inside (batch_)test_script_importing_module(.py)")
    print("")

    print(">from xmippPyModules import example_module")
    from xmippPyModules import example_module
    print("")

    print(">example_module.anyFunction()")
    print(example_module.anyFunction())
    print("")

    print(">cls = example_module.anyClass")
    cls = example_module.anyClass
    print("")

    print(">cls.getFromClassMethod()")
    print(cls.getFromClassMethod())
    print("")

    print(">obj = cls()")
    obj = cls()
    print("")

    print(">obj.getFromObjectMethod()")
    print(obj.getFromObjectMethod())
    print("")


    print(">from xmippPyModules.example_module2 import example_inmodule2")
    from xmippPyModules.example_module2 import example_inmodule2
    print("")

    print(">example_inmodule.anyFunction2()")
    print(example_inmodule2.anyFunction2())
    print("")

    print(">cls2 = example_module.anyClass")
    cls2 = example_inmodule2.anyClass2
    print("")

    print(">cls2.getFromClassMethod2()")
    print(cls2.getFromClassMethod2())
    print("")

    print(">obj2 = cls2()")
    obj2 = cls2()
    print("")

    print(">obj2.getFromObjectMethod2()")
    print(obj2.getFromObjectMethod2())
    print("")

except Exception as e:
    print(e)
    print(green("[==========]") + " 1 test ")
    print(  red("[  FAILED  ]") + " 1 test. ")
    sys.exit(1)


print(green("[==========]")+" 1 test ")
print(green("[  PASSED  ]")+" 1 test. ")

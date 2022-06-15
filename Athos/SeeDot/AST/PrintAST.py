"""

Authors: Sridhar Gopinath, Nishant Kumar.

Copyright:
Copyright (c) 2020 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

"""

from numpy import save
import AST.AST as AST
from AST.ASTVisitor import ASTVisitor
import binascii
import builtins

indent = ""

saveFile = open("ast.txt", "w")


# Overrides builtin print to force write to file.
# Caveat is that end *needs* to be specified, else "None" is printed
def print(*args, sep=None, end=None):
    for arg in args:
        builtins.print(arg, sep="", end="", file=saveFile, flush=True)

    builtins.print(end, sep="", end="", file=saveFile, flush=True)


# Prints list as comma separated string
def liststr(data):

    if data is None:
        return " "

    data = [str(elem) for elem in data]
    return ", ".join(data)


class PrintAST(ASTVisitor):

    # Value info contains shapes of the variables
    def __init__(self, value_info) -> None:
        self.value_info = value_info
        super().__init__()

    # TODO : fix printing of AST
    def visitInt(self, node: AST.Int, args=None):
        print(indent * node.depth, node.value, end=" ")

    def visitFloat(self, node: AST.Float, args=None):
        print(indent * node.depth, node.value, end=" ")

    def visitId(self, node: AST.ID, args=None):
        # print("{Called node.id}", end='')
        print(node.name, end=" ")

    def visitDecl(self, node: AST.Decl, args=None):
        if node.valueList:
            print(
                indent * node.depth,
                liststr(node.shape),
                list(map(lambda x: x.value, node.valueList)),
                end=" ",
            )
        else:
            print(indent * node.depth, liststr(node.shape), end=" ")

    def visitTranspose(self, node: AST.Transpose, args=None):
        node.expr.depth = node.depth + 1
        # print(node.__dict__, end="\n")
        print("Transpose(", end=" ")
        self.visit(node.expr)
        print(", ", end="")

    def visitSlice(self, node: AST.Transpose, args=None):
        node.expr.depth = node.depth + 1
        print(indent * node.depth, end=" ")
        self.visit(node.expr)
        print("extract slice", end=" ")

    def visitReshape(self, node: AST.Reshape, args=None):
        node.expr.depth = node.depth + 1
        orderLen = "" if node.order is None else str(len(node.order))
        print(f"reshape{orderLen}(", end=" ")
        self.visit(node.expr)
        if node.order:
            print(", ", liststr(node.shape), ", ", liststr(node.order), end=", ")
        else:
            print(liststr(node.shape), end=", ")

    def visitGather(self, node: AST.Gather, args=None):
        node.expr.depth = node.depth + 1
        print(indent * node.depth, "Gather", end=" ")
        self.visit(node.expr)
        print(liststr(node.shape), node.axis, node.index, end=" ")

    def visitPool(self, node: AST.Pool, args=None):
        node.expr.depth = node.depth + 1
        print(indent * node.depth, node.poolType, end=" ")
        self.visit(node.expr)

    def visitUOp(self, node: AST.UOp, args=None):
        node.expr.depth = node.depth + 1
        print(indent * node.depth, AST.OperatorsSymbolDict[node.op.name], end=" ")
        self.visit(node.expr)

    # function(operator 1, operator 2, options, shape, output)
    def visitBOp(self, node: AST.BOp, args=None):
        node.expr1.depth = node.expr2.depth = node.depth + 1
        print(f"{node.op.name}( ", end="")
        self.visit(node.expr1)
        print(", ", end="")
        # print(AST.OperatorsSymbolDict[node.op.name], end=" ")
        self.visit(node.expr2)
        print(", ", end="")
        if node.options is not None:
            print(liststr(node.options.values()), end=", ")
        if node.type is not None:
            print(liststr(node.type.shape), end=", ")

    def visitFunc(self, node: AST.Func, args=None):
        print(indent * node.depth, AST.OperatorsSymbolDict[node.op.name], end="(")
        node.expr.depth = node.depth + 1
        self.visit(node.expr)
        print(", ", end="")

    def visitLinearLayer(self, node: AST.BOp, args=None):
        print("LinearLayer(", end=" ")
        self.visit(node.expr1.expr1)
        print(", ", end="")
        print(liststr(node.expr1.expr1.type.shape), end=", ")
        print(node.expr)

    def visitLet(self, node: AST.Let, args=None):

        # Check for LinearLayer expression ( y = w*x_transpose + b )
        try:
            if node.decl.expr1.op == AST.Operators.MUL and isinstance(
                node.decl.expr1.expr2, AST.Transpose
            ):
                # Now we are in LinearLayer
                self.visitLinearLayer(node.decl)
                self.visit(node.expr)
        except:
            self.visit(node.decl)
            print(node.name.name, end="")
            print(")", end="")
            print("", end="\n")
            self.visit(node.expr)

    def visitUninterpFuncCall(self, node: AST.UninterpFuncCall, args=None):
        print("UninterpFuncCall( ", node.funcName, end=", ")
        for x in node.argsList:
            self.visit(x)
            print(", ", end="")

    def visitArgMax(self, node: AST.ArgMax, args=None):
        print("ArgMax(", end=" ")
        self.visit(node.expr)
        self.visit(node.dim)

    def visitReduce(self, node: AST.Reduce, args=None):
        print(
            indent * node.depth,
            "reduce",
            AST.OperatorsSymbolDict[node.op.name],
            end=" ",
        )
        self.visit(node.expr)

    def visitInput(self, node: AST.Input, args=None):
        print(
            indent * node.depth,
            "input( ",
            liststr(node.shape),
            ", ",
            node.dataType,
            ", ",
            node.inputByParty.name,
            ", ",
            end="",
        )

    def visitOutput(self, node: AST.Output, args=None):
        print(indent * node.depth, "output( ", end="")
        node.expr.depth = node.depth + 1
        self.visit(node.expr)
        print(indent * node.depth, ")", end="")

    def visitFusedBatchNorm(self, node: AST.FusedBatchNorm, args=None):
        node.expr.depth = node.multExpr.depth = node.addExpr.depth = node.depth + 1
        print(indent * node.depth, "FusedBatchNorm", end=" ")
        self.visit(node.expr)
        self.visit(node.multExpr)
        self.visit(node.addExpr)

import re, copy, math, time, numpy, random

class Piece():
    def __init__(self,t,colour,x,y):
        self.type = t
        self.colour = colour
        self.file = x
        self.rank = 7-y
        self.pList = ['P','N','B','R','Q','K']
        self.taken = False

    def printPiece(self):
        print(self.pList[self.type])

class Board():
    def __init__(self):
        self.board = [[0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0],
                      [0,0,0,0,0,0,0,0]]
        
        self.setupBoard()
        self.pieces = []
        for i in self.board:
            for piece in i:
                if piece != 0:
                    self.pieces.append(piece)
    def setupBoard(self):
        pOrder = [4,2,3,5,6,3,2,4]; colour = 1
        for y in range(8):
            for x in range(8):
                if y > 4:
                    colour = 0
                if y == 0 or y == 7:
                    self.board[y][x] = Piece(pOrder[x],colour,x,y)
                if y == 1 or y == 6:
                    self.board[y][x] = Piece(1,colour,x,y)

    def printBoard(self):
        for line in self.board:
            row = []
            for piece in line:
                if piece != 0:
                    row.append(piece.pList[piece.type-1])
                else:
                    row.append("0")
            print(' '.join(row))

    def applyMove(self,move):
        if move.piece != 7:
            dest = [ord(move.dest[0])-97,int(move.dest[1])-1]
            piece = self.findPiece(move.piece,dest,move.txt,move.colour)

            if move.piece == 1 and "x" in move.txt and self.board[7-dest[1]][dest[0]] == 0:
                if piece.colour == 0:
                    self.removePiece([dest[0],dest[1]+1])
                    self.board[7-dest[1]+1][dest[0]] = 0
                else:
                    self.removePiece([dest[0],dest[1]-1])
                    self.board[7-dest[1]-1][dest[0]] = 0

            piecePos = [7-piece.rank,piece.file]
            pieceDest = [7-dest[1],dest[0]]

            self.board[7-piece.rank][piece.file] = 0
            self.board[7-dest[1]][dest[0]] = piece
            piece.file = dest[0]
            piece.rank = dest[1]

            if "=" in move.txt:
                piece.type = (move.pList.index(move.txt[-1])) + 2
        else:
            if len(move.txt) == 3:
                cPieces = []
                for i in self.pieces:
                    if i.type == 6 and i.colour == move.colour:
                        cPieces.append(i)
                for i in self.pieces:
                    if i.type == 4 and i.file == 7 and i.colour == move.colour:
                        if i.colour == 0 and i.rank == 0:
                            cPieces.append(i)
                        if i.colour == 1 and i.rank == 7:
                            cPieces.append(i)

                piecePos = [7-cPieces[0].rank,cPieces[0].file]
                pieceDest = [7-cPieces[0].rank,cPieces[0].file+2]

                self.board[7-cPieces[0].rank][cPieces[0].file] = 0
                self.board[7-cPieces[0].rank][cPieces[0].file+2] = cPieces[0]

                self.board[7-cPieces[1].rank][cPieces[1].file] = 0
                self.board[7-cPieces[1].rank][cPieces[1].file-2] = cPieces[1]

                cPieces[0].file = cPieces[0].file+2
                cPieces[1].file = cPieces[1].file-2

            else:
                cPieces = []
                for i in self.pieces:
                    if i.type == 6 and i.colour == move.colour:
                        cPieces.append(i)
                for i in self.pieces:
                    if i.type == 4 and i.file == 0 and i.colour == move.colour:
                        if i.colour == 0 and i.rank == 0:
                            cPieces.append(i)
                        if i.colour == 1 and i.rank == 7:
                            cPieces.append(i)

                piecePos = [7-cPieces[0].rank,cPieces[0].file]
                pieceDest = [7-cPieces[0].rank,cPieces[0].file-2]
                        
                self.board[7-cPieces[0].rank][cPieces[0].file] = 0
                self.board[7-cPieces[0].rank][cPieces[0].file-2] = cPieces[0]

                self.board[7-cPieces[1].rank][cPieces[1].file] = 0
                self.board[7-cPieces[1].rank][cPieces[1].file+3] = cPieces[1]

                cPieces[0].file = cPieces[0].file-2
                cPieces[1].file = cPieces[1].file+3

        return piecePos, pieceDest

    def findPiece(self,t,dest,pgn,colour):
        possPieces = []
        
        for piece in self.pieces:
            if piece.type == t and piece.colour == colour and piece.taken == False:
                diff = [abs(piece.file-dest[0]),abs((piece.rank)-dest[1])]
                if t == 1 and ((piece.colour == 0 and (piece.rank-dest[1]) < 0) or (piece.colour == 1 and (piece.rank-dest[1]) > 0)):
                    if "x" not in pgn:
                        if piece.file == dest[0]:
                                possPieces.append(piece)
                            
                    if "x" in pgn:
                        if diff == [1,1] and ord(pgn[0])-97 == piece.file:
                                possPieces.append(piece)
                        
                if t == 2:
                    if diff == [1,2] or diff == [2,1]:
                        possPieces.append(piece)
                        
                if t == 3:
                    possDiffs = [[i,i] for i in range(1,9)]
                    if diff in possDiffs:
                        if self.checkCanReach(piece,dest,pgn):
                            possPieces.append(piece)

                if t == 4:
                    possDiffs2 = [[i,0] for i in range(1,9)]
                    possDiffs3 = [[0,i] for i in range(1,9)]
                    if diff in possDiffs2 or diff in possDiffs3:
                        if self.checkCanReach(piece,dest,pgn):
                            possPieces.append(piece)

                if t == 5:
                    possDiffs = [[i,i] for i in range(1,9)]
                    possDiffs2 = [[i,0] for i in range(1,9)]
                    possDiffs3 = [[0,i] for i in range(1,9)]
                    if diff in possDiffs or diff in possDiffs2 or diff in possDiffs3:
                        if self.checkCanReach(piece,dest,pgn):
                            possPieces.append(piece)

                if t == 6:
                    possDiffs = [[1,0],[0,1],[1,1]]
                    if diff in possDiffs:
                        possPieces.append(piece)

                if "x" in pgn:
                    self.removePiece(dest)

        if len(possPieces) > 1:
            nPossPieces = []
            reqRank = 8
            reqFile = 8
            check = 1
            if (len(pgn) == 4 or (len(pgn) == 5 and "x" in pgn)) and "=" not in pgn:
                try:
                    if t == 1:
                        check = 0 
                    if pgn[check] in ['a','b','c','d','e','f','g','h']:
                        reqFile = ord(pgn[check])-97
                    else:
                        reqRank = int(pgn[check]) - 1

                    for i in possPieces:
                        if i.file == reqFile:
                            nPossPieces.append(i)
                        elif i.rank == reqRank:
                            nPossPieces.append(i)

                except:
                    pass
                    
            if t == 1 and nPossPieces == []:
                shortestDist = 8
                closestPiece = 0
                for i in possPieces:
                    if abs(i.rank-dest[1]) < shortestDist:
                        shortestDist = abs(i.rank-dest[1])
                        closestPiece = i
                nPossPieces.append(closestPiece)
                
            if nPossPieces == []:
                filt = False
                for i in possPieces:
                    check = self.checkRevealedMate(i,dest)
                    #print(check)
                    if check == True:
                        filt = True
                        
                if filt == True:
                     for i in possPieces:
                        check = self.checkRevealedMate(i,dest)
                        if check == False:
                            nPossPieces.append(i)

            p = [nPossPieces[0]]
            for i in nPossPieces:
                for j in p:
                    if i.file != j.file and i.rank != j.rank:
                        p.append(i)
            nPossPieces = p
            possPieces = nPossPieces

        if len(possPieces) > 1:
            print("DISAMBIGUATION NEEDED")
            possPieces = []
        if len(possPieces) == 0:
            pass

        return possPieces[0]

    def removePiece(self,dest):
        for i in self.pieces:
            if i.file == dest[0] and i.rank == dest[1]:
                i.taken = True

    def checkRevealedMate(self,piece,dest):
        cBoard = copy.deepcopy(self)
        cBoard.board[7-piece.rank][piece.file] = 0
        cBoard.board[7-dest[1]][dest[0]] = piece

        inCheck = cBoard.checkCheck(piece.colour,dest)
        return inCheck

    def checkCheck(self,colour,dest):
        for i in range(8):
            for j in range(8):
                if self.board[j][i] != 0:
                    if self.board[j][i].type == 6 and self.board[j][i].colour == colour:
                        kingPos = [i,7-j]
        
        diffs = [[[1,1]],
                 [[1,2],[2,1]],
                 [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]],
                 [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0]],
                 [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]]

        for piece in self.pieces:
            if piece.taken == False and piece.colour != colour and piece.type != 6 and dest != [piece.file,piece.rank]:
                diff = [abs(piece.file-kingPos[0]),abs((piece.rank)-kingPos[1])]
                if diff in diffs[piece.type-1]:
                    checkPos = [piece.file,piece.rank]
                    nDiff = [(piece.file-kingPos[0]),((piece.rank)-kingPos[1])]
                    delta = [(nDiff[0]/max(diff))*-1,(nDiff[1]/max(diff))*-1]
                    for i in range(1,max(diff)+1):
                        checkPos[0] = int(checkPos[0] + delta[0])
                        checkPos[1] = int(checkPos[1] + delta[1])
                        if checkPos == kingPos:
                            return True
                        if self.board[7-int(checkPos[1])][int(checkPos[0])] != 0:
                            break
        return False

    def checkCanReach(self,piece,dest,pgn):
        diffs = [[[1,1]],
         [[1,2],[2,1]],
         [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]],
         [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0]],
         [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]]

        diff = [abs(piece.file-dest[0]),abs((piece.rank)-dest[1])]
        if diff in diffs[piece.type-1]:
            checkPos = [piece.file,piece.rank]
            nDiff = [(piece.file-dest[0]),((piece.rank)-dest[1])]
            delta = [(nDiff[0]/max(diff))*-1,(nDiff[1]/max(diff))*-1]

            for i in range(1,max(diff)+1):
                checkPos[0] = int(checkPos[0] + delta[0])
                checkPos[1] = int(checkPos[1] + delta[1])
                if checkPos == dest:
                    return True
                if self.board[7-int(checkPos[1])][int(checkPos[0])] != 0:
                    break

        return False         
                    
class Move():
    def __init__(self,pgn, colour):
        self.pList = ['N','B','R','Q','K']
        self.txt = pgn
        self.piece = self.getPiece()
        self.colour = colour

        if "=" not in self.txt:
            self.dest = pgn[-2:]
        else:
            self.dest = pgn[-4:-2]

    def getPiece(self):
        if self.txt[0] not in self.pList:
            if self.txt[0] == "O":
                p = 7
            else:
                p = 1
        else:
            p = (self.pList.index(self.txt[0])) + 2 
        return p

def cC(colour,dest,board):
    for i in range(8):
        for j in range(8):
            if board.board[j][i] != 0:
                if board.board[j][i].type == 6 and board.board[j][i].colour == colour:
                    kingPos = [i,7-j]

    diffs = [[[1,1]],
             [[1,2],[2,1]],
             [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]],
             [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0]],
             [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]]]

    for piece in board.pieces:
        if piece.taken == False and piece.colour != colour and piece.type != 6 and dest != [piece.file,piece.rank]:
            try:
                diff = [abs(piece.file-kingPos[0]),abs((piece.rank)-kingPos[1])]
            except:
                board.printBoard()
                print(kingPos)
            if diff in diffs[piece.type-1]:
                checkPos = [piece.file,piece.rank]
                nDiff = [(piece.file-kingPos[0]),((piece.rank)-kingPos[1])]
                delta = [(nDiff[0]/max(diff))*-1,(nDiff[1]/max(diff))*-1]
                for i in range(1,max(diff)+1):
                    checkPos[0] = int(checkPos[0] + delta[0])
                    checkPos[1] = int(checkPos[1] + delta[1])
                    if checkPos == kingPos:
                        return True
                    if board.board[7-int(checkPos[1])][int(checkPos[0])] != 0:
                        break
    return False

def reverseExF(array):
    nArray = []; counter = 0
    for i in range(8):
        iA = []
        for j in range(8):
            jA = []
            for k in range(8):
                kA = []
                for l in range(8):
                    kA.append(array[counter])
                    counter += 1
                jA.append(kA)
            iA.append(jA)
        nArray.append(iA)

    return nArray

def pBoard(b):
    for line in b:
        row = []
        for piece in line:
            if piece != 0:
                row.append(piece.pList[piece.type-1])
            else:
                row.append("0")
        print(' '.join(row))
            
def parseText(pgn,board):
    moves = []; boardData = [copy.deepcopy(board.board)]; pPoses = []; pDests = []; boards = [copy.deepcopy(board)]
    data = pgn.split()
    
    for mov in data:
        if mov not in ["1-0","0-1","1/2-1/2","*"]:
            if mov[len(mov)-1]=="+" or mov[len(mov)-1]=="#":
                mov = mov[:-1]
            r = re.sub('\d+[\.]',"",mov)
            if r != "":
                moves.append(r)
        else:
            winPGN = mov

    if winPGN == "1-0":
        winner = 0
    if winPGN == "0-1":
        winner = 1
    if winPGN == "1/2-1/2" or winPGN == "*":
        winner = 2

    gMoves = []
    for i in range(len(moves)):
        gMoves.append(Move(moves[i],(i+2)%2))
    for i in range(len(gMoves)):
        pPos, pDest = board.applyMove(gMoves[i])
        boardData.append(copy.deepcopy(board.board))
        pPoses.append(pPos); pDests.append(pDest)

    whiteCMs = []; blackCMs = []
    for i in boardData:
        whiteCMs.append(createCM(i,0))
        blackCMs.append(createCM(i,1))
        
    return boardData, pPoses, pDests, winner, whiteCMs, blackCMs

def createCM(board,c):
    coverMap = numpy.zeros((8,8))
    for rank in range(8):
        for file in range(8):
            piece = board[rank][file]
            if piece != 0:
                if piece.colour == c:
                    possMoves = getMoves(piece.type,rank,file,board,c)
                    for i in possMoves:
                        coverMap[7-i[1]][i[0]] = 1
    return coverMap

def pBoard(board):
    for line in board:
        row = []
        for piece in line:
            if piece != 0:
                row.append(piece.pList[piece.type-1])
            else:
                row.append("0")
        print(' '.join(row))


def getMoves(t,rank,file,board,c):
    diffs = [[[1,1]],
     [[1,2],[2,1]],
     [[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]],
     [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0]],
     [[0,1],[0,2],[0,3],[0,4],[0,5],[0,6],[0,7],[1,0],[2,0],[3,0],[4,0],[5,0],[6,0],[7,0],[1,1],[2,2],[3,3],[4,4],[5,5],[6,6],[7,7]],
    [[1,1],[1,0],[0,1]]]

    pawnDiffs = [[[-1,1],[1,1]],[[-1,-1,],[1,-1]]]

    p = [[1,1],[1,-1],[-1,1],[-1,-1]]
    pos = [file,7-rank]
    moves = []; doneDirections = []
    #print(pos)
    for i in (diffs[t-1]):
        if t != 1:
            for k in range(len(p)):
                dest = [pos[0]+(i[0]*p[k][0]),(pos[1]+(i[1]*p[k][1]))]
                direction = [(dest[0]-pos[0])/max([1,abs(dest[0]-pos[0])]),(dest[1]-pos[1])/max([1,abs(dest[1]-pos[1])])]
                if dest[0] < 8 and dest[0] >= 0 and dest[1]<8 and dest[1]>=0:
                    if board[7-dest[1]][dest[0]] == 0 and direction not in doneDirections:
                        moves.append(dest)
                    if board[7-dest[1]][dest[0]] != 0 and direction not in doneDirections:
                        moves.append(dest)
                        if t != 2:
                            doneDirections.append(direction)                      

        else:
            for i in pawnDiffs[c]:
                dest = [pos[0]+i[0],pos[1]+i[1]]
                if dest[0] < 8 and dest[0] >= 0 and dest[1]<8 and dest[1]>=0:
                    moves.append(dest)

    return moves
                

def prepData(game,w,wcms,bcms):
    inp = []
    out = []

    for i in range(1,len(game)):
        if i%2 == 0:
            c = 1
        else:
            c = 0

        if w == c or w == 2:
            ot = dim(game[i])
            ip = dim(game[i-1])

            for j in range(len(ot)):
                for k in range(len(ot[j])):
                    ip[j][k].append(c)
                    ip[j][k].append(wcms[i-1][j][k])
                    ip[j][k].append(bcms[i-1][j][k])

            randN = random.random()   
            ip = f(ip)
            inp.append(ip)

    randN = random.random()
    half = False
    if randN > 1:
        inp = inp[int(len(inp)/2):]
        half = True

    return inp, half

def dim(turn):
    array = [[[] for i in range(8)] for j in range(8)]

    for i in range(len(turn)):
        for j in range(len(turn[i])):
            newArray = [0 for i in range(8)]

            if turn[i][j] == 0:
                newArray[0] = 1
            else:
                newArray[turn[i][j].type] = 1
                newArray[7] = turn[i][j].colour
                

            array[i][j] = newArray

    return array

def exEvalMove(array,board,iMoves):
    topMove = 0
    index = [0,0,0,0]
    for i in range(len(array)):
        for j in range(len(array[0])):
            for k in range(len(array[0][0])):
                for l in range(len(array[0][0][0])):
                    if array[i][j][k][l] > topMove:
                        if board.board[i][j] != 0 and [i,j,k,l] not in iMoves:
                            topMove = array[i][j][k][l]
                            index = [i,j,k,l]

    return index,topMove

def mutEx(pos,dest,w, half):
    finalArray = []; c = 0

    for i in range(len(pos)):
        if w == c or w == 2:
            array = numpy.zeros((8,8,8,8))
            array[pos[i][0]][pos[i][1]][dest[i][0]][dest[i][1]] = 1
            finalArray.append(array.flatten())

        c = (c*-1) + 1
        
    if half == True:
        finalArray = finalArray[int(len(finalArray)/2):]
        
    return finalArray

% Ajedrez en Prolog: motor básico
% - Representación: FEN + tablero de 64 celdas (a8..h1), índice 0..63
% - Movimientos: todas las piezas; sin enroque ni paso; promoción a dama
% - Legalidad: filtra por jaque
% - Evaluación simple: material + movilidad (pesos en weights.pl)

:- initialization(init_chess).
:- use_module(library(lists)).  % nth0/3, append/3, sum_list/2, same_length/2
:- use_module(library(apply)).  % include/3, maplist/2,3
:- discontiguous gen_piece_move/5.

init_chess :-
    catch(consult('prolog/weights.pl'), _, true).

% Helpers de CLI para Haskell (imprimen resultado plano)
cli_best_move(FEN) :-
    best_move(FEN, Move), !,
    ( Move = none -> writeln(none)
    ; writeln(Move)
    ).
cli_best_move(_) :- writeln(none).

cli_apply_move(FEN, Move) :-
    ( apply_move(FEN, Move, NewFEN) -> writeln(NewFEN)
    ; writeln(illegal)
    ).

cli_is_legal(FEN, Move) :-
    ( legal_move(FEN, Move) -> writeln(true)
    ; writeln(false)
    ).

cli_status(FEN) :-
    status(FEN, S), writeln(S).

% generar tablero 

initial_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w - - 0 1').

fen_board(FEN, Board, Side) :-
    split_string(FEN, " ", " ", [PieceField, SideField | _]),
    side_from_field(SideField, Side),
    board_from_piece_field(PieceField, Board).

side_from_field("w", w).
side_from_field("b", b).

board_from_piece_field(Field, Board) :-
    split_string(Field, "/", "", Ranks),
    maplist(rank_to_cells, Ranks, CellLists),
    append(CellLists, Board).

rank_to_cells(RankStr, Cells) :-
    string_chars(RankStr, Cs),
    rank_chars_to_cells(Cs, Cells).

rank_chars_to_cells([], []).
rank_chars_to_cells([C|Cs], Cells) :-
    char_type(C, digit), !,
    atom_number(C, N),
    blanks(N, Bs),
    rank_chars_to_cells(Cs, Rest),
    append(Bs, Rest, Cells).
rank_chars_to_cells([C|Cs], [P|Rest]) :-
    piece_from_char(C, P),
    rank_chars_to_cells(Cs, Rest).

blanks(0, []).
blanks(N, [empty|T]) :- N > 0, N1 is N-1, blanks(N1, T).

piece_from_char('P', piece(w,p)).
piece_from_char('N', piece(w,n)).
piece_from_char('B', piece(w,b)).
piece_from_char('R', piece(w,r)).
piece_from_char('Q', piece(w,q)).
piece_from_char('K', piece(w,k)).
piece_from_char('p', piece(b,p)).
piece_from_char('n', piece(b,n)).
piece_from_char('b', piece(b,b)).
piece_from_char('r', piece(b,r)).
piece_from_char('q', piece(b,q)).
piece_from_char('k', piece(b,k)).

char_from_piece(piece(w,p), 'P').
char_from_piece(piece(w,n), 'N').
char_from_piece(piece(w,b), 'B').
char_from_piece(piece(w,r), 'R').
char_from_piece(piece(w,q), 'Q').
char_from_piece(piece(w,k), 'K').
char_from_piece(piece(b,p), 'p').
char_from_piece(piece(b,n), 'n').
char_from_piece(piece(b,b), 'b').
char_from_piece(piece(b,r), 'r').
char_from_piece(piece(b,q), 'q').
char_from_piece(piece(b,k), 'k').

board_to_piece_field(Board, Field) :-
    board_to_ranks(Board, Ranks),
    maplist(rank_cells_to_string, Ranks, RankStrs),
    atomic_list_concat(RankStrs, '/', Field).

board_to_ranks(Board, [A,B,C,D,E,F,G,H]) :-
    length(Board, 64),
    append(A, Rest1, Board), length(A,8),
    append(B, Rest2, Rest1), length(B,8),
    append(C, Rest3, Rest2), length(C,8),
    append(D, Rest4, Rest3), length(D,8),
    append(E, Rest5, Rest4), length(E,8),
    append(F, Rest6, Rest5), length(F,8),
    append(G, H, Rest6),    length(G,8), length(H,8).

rank_cells_to_string(Cells, Str) :-
    rank_cells_to_parts(Cells, Parts),
    atomic_list_concat(Parts, '', Str).

rank_cells_to_parts([], []).
rank_cells_to_parts([empty|Cs], Parts) :-
    count_blanks(Cs, N, Rest), N1 is N+1, number_string(N1, S), atom_string(A,S),
    rank_cells_to_parts(Rest, PartsRest), Parts = [A|PartsRest].
rank_cells_to_parts([piece(C,T)|Cs], [X|Rest]) :-
    char_from_piece(piece(C,T), X),
    rank_cells_to_parts(Cs, Rest).

count_blanks([empty|Cs], N, Rest) :- !, count_blanks(Cs, N1, Rest), N is N1+1.
count_blanks(Rest, 0, Rest).

toggle(w,b). toggle(b,w).

fen_from(Board, Side, FEN) :-
    board_to_piece_field(Board, Field),
    (Side = w -> SideS = 'w' ; SideS = 'b'),
    atomic_list_concat([Field, ' ', SideS, ' - - 0 1'], '', FEN).

% Move utilities 

idx(R,C,Idx) :- Idx is R*8 + C.
row(Idx,R) :- R is Idx // 8.
col(Idx,C) :- C is Idx mod 8.
on_board(Idx) :- Idx >= 0, Idx < 64.

at(Board, Idx, Val) :- nth0(Idx, Board, Val).
set_at(Board, Idx, Val, New) :- same_length(Board, New), nth0(Idx, New, Val, Rest), nth0(Idx, Board, _, Rest).

enemy(w,b). enemy(b,w).
piece_color(piece(C,_), C).
piece_type(piece(_,T), T).

square_name(Idx, Name) :- col(Idx,C), row(Idx,R),
    File is 0'a + C, atom_codes(F,[File]), Rank is 8 - R, number_string(Rank,RS), atom_string(RA,RS), atom_concat(F, RA, Name).

uci_to_idxs(Uci, From, To, Promo) :-
    atom_chars(Uci, Chars),
    ( Chars = [Ff,Fr,Tf,Tr,Pc] -> promo_char(Pc, Promo),
        file_rank_to_idx(Ff,Fr,From), file_rank_to_idx(Tf,Tr,To)
    ; Chars = [Ff,Fr,Tf,Tr] -> Promo = none,
        file_rank_to_idx(Ff,Fr,From), file_rank_to_idx(Tf,Tr,To)
    ).

promo_char('q', q). promo_char('r', r). promo_char('b', b). promo_char('n', n).

file_rank_to_idx(Ff,Fr,Idx) :-
    char_code(Ff, FC), C is FC - 0'a,
    number_chars(NR,[Fr]), R is 8 - NR,
    idx(R,C,Idx).

idxs_to_uci(From, To, Promo, Uci) :-
    square_name(From, A), square_name(To, B),
    ( Promo = none -> atom_concat(A,B,Uci)
    ; promo_atom(Promo,Pa), atom_concat(A,B,AB), atom_concat(AB,Pa,Uci)
    ).

promo_atom(q,'q'). promo_atom(r,'r'). promo_atom(b,'b'). promo_atom(n,'n').

%  Move generation

gen_pseudo_moves(Board, Side, Moves) :-
    findall(m(F,T,none), gen_piece_move(Board, Side, F, T, none), M1),
    findall(m(F,T,P), gen_pawn_promo_move(Board, Side, F, T, P), M2),
    append(M1, M2, Moves).

gen_piece_move(Board, Side, From, To, none) :-
    nth0(From, Board, piece(Side, Type)), Type \= p,
    ( Type = n -> knight_moves(Board, Side, From, To)
    ; Type = k -> king_moves(Board, Side, From, To)
    ; Type = b -> slide_moves(Board, Side, From, To, [-9,-7,7,9])
    ; Type = r -> slide_moves(Board, Side, From, To, [-8,8,-1,1])
    ; Type = q -> slide_moves(Board, Side, From, To, [-9,-7,7,9,-8,8,-1,1])
    ).

% Sliding pieces
slide_moves(Board, Side, From, To, Deltas) :- member(D, Deltas), slide_dir(Board, Side, From, D, To).

slide_dir(Board, Side, From, D, To) :-
    step_slide(Board, Side, From, D, From, To).

step_slide(Board, Side, Orig, D, Cur, To) :-
    next_in_dir(Cur, D, Next), on_board(Next),
    ( at(Board, Next, empty) -> (To = Next ; step_slide(Board, Side, Orig, D, Next, To))
    ; at(Board, Next, piece(Other,_)), Other \= Side, To = Next
    ).

next_in_dir(Idx, D, Next) :-
    row(Idx,R), col(Idx,C),
    ( D = -8 -> R1 is R-1, C1 is C
    ; D = 8  -> R1 is R+1, C1 is C
    ; D = -1 -> R1 is R,   C1 is C-1
    ; D = 1  -> R1 is R,   C1 is C+1
    ; D = -9 -> R1 is R-1, C1 is C-1
    ; D = -7 -> R1 is R-1, C1 is C+1
    ; D = 7  -> R1 is R+1, C1 is C-1
    ; D = 9  -> R1 is R+1, C1 is C+1
    ),
    R1 >= 0, R1 < 8, C1 >= 0, C1 < 8,
    idx(R1,C1,Next).

% Knight
knight_moves(Board, Side, From, To) :-
    row(From,R), col(From,C),
    member((DR,DC), [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]),
    R1 is R+DR, C1 is C+DC, R1>=0,R1<8,C1>=0,C1<8,
    idx(R1,C1,To),
    ( at(Board, To, empty) ; at(Board, To, piece(Other,_)), Other \= Side ).

% King (sin enroque)
king_moves(Board, Side, From, To) :-
    row(From,R), col(From,C),
    member((DR,DC), [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]),
    R1 is R+DR, C1 is C+DC, R1>=0,R1<8,C1>=0,C1<8,
    idx(R1,C1,To),
    ( at(Board, To, empty) ; at(Board, To, piece(Other,_)), Other \= Side ).

% Pawns
gen_pawn_promo_move(Board, Side, From, To, Promo) :-
    nth0(From, Board, piece(Side,p)),
    pawn_moves(Board, Side, From, To, Promo),
    Promo \= none.

gen_piece_move(Board, Side, From, To, none) :-
    nth0(From, Board, piece(Side,p)),
    pawn_moves(Board, Side, From, To, none).

pawn_moves(Board, w, From, To, Promo) :-
    row(From,R), col(From,C),
    R1 is R-1, R1>=0,
    % forward one
    idx(R1,C,One), ( at(Board, One, empty) ->
        ( (R1 =:= 0) -> Promo = q, To = One ; Promo = none, To = One )
    ; fail ),
    From \= To.
pawn_moves(Board, w, From, To, Promo) :-
    row(From,R), col(From,C), R=:=6, % double from rank 2
    R1 is R-1, R2 is R-2,
    idx(R1,C,One), idx(R2,C,Two), at(Board, One, empty), at(Board, Two, empty),
    Promo = none, To = Two.
pawn_moves(Board, w, From, To, Promo) :- % captures
    row(From,R), col(From,C), R1 is R-1,
    member(DC, [-1,1]), C1 is C+DC, R1>=0, C1>=0, C1<8,
    idx(R1,C1,To), at(Board, To, piece(b,_)),
    ( R1 =:= 0 -> Promo = q ; Promo = none ).

pawn_moves(Board, b, From, To, Promo) :-
    row(From,R), col(From,C),
    R1 is R+1, R1<8,
    idx(R1,C,One), ( at(Board, One, empty) ->
        ( (R1 =:= 7) -> Promo = q, To = One ; Promo = none, To = One )
    ; fail ),
    From \= To.
pawn_moves(Board, b, From, To, Promo) :-
    row(From,R), col(From,C), R=:=1, % double from rank 7
    R1 is R+1, R2 is R+2,
    idx(R1,C,One), idx(R2,C,Two), at(Board, One, empty), at(Board, Two, empty),
    Promo = none, To = Two.
pawn_moves(Board, b, From, To, Promo) :- % captures
    row(From,R), col(From,C), R1 is R+1,
    member(DC, [-1,1]), C1 is C+DC, R1<8, C1>=0, C1<8,
    idx(R1,C1,To), at(Board, To, piece(w,_)),
    ( R1 =:= 7 -> Promo = q ; Promo = none ).

%  Hacke 

king_index(Board, Side, Idx) :- nth0(Idx, Board, piece(Side,k)).

% Ataques a una casilla
square_attacked(Board, Idx, BySide) :-
    % Knights attacking
    row(Idx,R), col(Idx,C),
    member((DR,DC), [(-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)]),
    R1 is R+DR, C1 is C+DC, R1>=0,R1<8,C1>=0,C1<8, idx(R1,C1,N), at(Board, N, piece(BySide,n)).
square_attacked(Board, Idx, BySide) :-
    % King attacking
    row(Idx,R), col(Idx,C),
    member((DR,DC), [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]),
    R1 is R+DR, C1 is C+DC, R1>=0,R1<8,C1>=0,C1<8, idx(R1,C1,N), at(Board, N, piece(BySide,k)).
square_attacked(Board, Idx, BySide) :-
    % Rooks/Queens along ranks/files
    member(D, [-8,8,-1,1]), ray_attacks(Board, Idx, D, BySide, [r,q]).
square_attacked(Board, Idx, BySide) :-
    % Bishops/Queens along diagonals
    member(D, [-9,-7,7,9]), ray_attacks(Board, Idx, D, BySide, [b,q]).
square_attacked(Board, Idx, w) :- % white pawn attacks upwards
    row(Idx,R), col(Idx,C), R<7,
    member(DC, [-1,1]), R1 is R+1, C1 is C+DC, R1<8,C1>=0,C1<8, idx(R1,C1,N), at(Board, N, piece(w,p)).
square_attacked(Board, Idx, b) :- % black pawn attacks downwards
    row(Idx,R), col(Idx,C), R>0,
    member(DC, [-1,1]), R1 is R-1, C1 is C+DC, R1>=0,C1>=0,C1<8, idx(R1,C1,N), at(Board, N, piece(b,p)).

ray_attacks(Board, Idx, D, BySide, Types) :-
    next_in_dir(Idx, D, Next), on_board(Next),
    ( at(Board, Next, empty) -> ray_attacks(Board, Next, D, BySide, Types)
    ; at(Board, Next, piece(BySide,T)), member(T, Types)
    ).

in_check(Board, Side) :- king_index(Board, Side, K), enemy(Side, Opp), square_attacked(Board, K, Opp).

gen_legal_moves(Board, Side, Moves) :-
    gen_pseudo_moves(Board, Side, Ms),
    include(is_legal_move(Board, Side), Ms, Moves).

is_legal_move(Board, Side, m(F,T,Promo)) :-
    attempt_move(Board, Side, m(F,T,Promo), NB),
    \+ in_check(NB, Side).

attempt_move(Board, _Side, m(F,T,Promo), NB) :-
    nth0(F, Board, piece(Side,Type)),
    ( Promo \= none -> Type1 = p ; Type1 = Type ),
    replace_move(Board, F, T, piece(Side,Type1), Promo, NB).

replace_move(Board, F, T, Piece, Promo, NB) :-
    set_at(Board, F, empty, B1),
    ( Promo = none -> FinalPiece = Piece ; piece_color(Piece,C), FinalPiece = piece(C,q) ),
    set_at(B1, T, FinalPiece, NB).

% Public API 

generate_legal_moves(FEN, MovesUci) :-
    fen_board(FEN, Board, Side),
    gen_legal_moves(Board, Side, Ms),
    maplist(move_to_uci, Ms, MovesUci).

move_to_uci(m(F,T,Promo), U) :- idxs_to_uci(F,T,Promo,U).

legal_move(FEN, Uci) :-
    uci_to_idxs(Uci, F, T, Promo),
    fen_board(FEN, Board, Side),
    gen_legal_moves(Board, Side, Ms),
    member(m(F,T,Promo), Ms).

apply_move(FEN, Uci, NewFEN) :-
    fen_board(FEN, Board, Side),
    uci_to_idxs(Uci, F, T, Promo),
    gen_legal_moves(Board, Side, Ms),
    member(m(F,T,Promo), Ms),
    attempt_move(Board, Side, m(F,T,Promo), NB),
    toggle(Side, Next),
    fen_from(NB, Next, NewFEN).

% Search 

best_move(FEN, MoveUci) :-
    fen_board(FEN, Board, Side),
    gen_legal_moves(Board, Side, Ms),
    ( Ms = [] -> MoveUci = none
    ; best_of(Board, Side, Ms, MoveUci)
    ).

best_of(Board, Side, Ms, BestUci) :-
    findall(ScoreAdj-U, (
        member(M, Ms),
        attempt_move(Board, Side, M, NB),
        toggle(Side, Opp),
        eval(NB, Opp, Score),
        side_factor(Side, F), ScoreAdj is F * Score,
        move_to_uci(M, U)
    ), Pairs),
    keysort(Pairs, Sorted), last_pair(Sorted, _-BestUci).

side_factor(w, 1). side_factor(b, -1).

last_pair([X], X).
last_pair([_|T], X) :- last_pair(T, X).

% Evaluation 

piece_value(p, 1). piece_value(n, 3). piece_value(b, 3). piece_value(r, 5). piece_value(q, 9). piece_value(k, 0).

eval(Board, SideToMove, Score) :-
    mat_score(Board, SM),
    mobility_score(Board, SideToMove, Mob),
    weight(material, WM), weight(mobility, WMO),
    Score is WM * SM + WMO * Mob.

mat_score(Board, S) :-
    findall(V, (member(piece(w,T), Board), piece_value(T,V)), WVals), sum_list(WVals, W),
    findall(V2,(member(piece(b,T2), Board), piece_value(T2,V2)), BVals), sum_list(BVals, B),
    S is W - B.

mobility_score(Board, SideToMove, S) :-
    gen_legal_moves(Board, SideToMove, Ms), length(Ms, N), S is N.

%  Learning

% Placeholder para actualizar pesos tras una partida. Se puede extender con
% una señal desde Haskell con el historial y el resultado.
learn_result(_History, _Result) :-
    true.

%  Status & Learning 

status(FEN, Status) :-
    fen_board(FEN, Board, Side),
    gen_legal_moves(Board, Side, Ms),
    ( Ms = [] -> ( in_check(Board, Side) -> Status = checkmate ; Status = stalemate )
    ; Status = ok
    ).

% Regla simple: ajustar peso de movilidad según resultado
% outcome: system_win | system_loss | draw
learn_simple(Outcome) :-
    ( weight(mobility, W0) -> true ; W0 = 1 ),
    ( Outcome = system_win  -> W1 is W0 + 0.2
    ; Outcome = system_loss -> W1 is max(0, W0 - 0.2)
    ; Outcome = draw        -> W1 is W0 + 0.0
    ),
    retractall(weight(mobility,_)), assertz(weight(mobility, W1)),
    save_weights.

save_weights :-
    open('prolog/weights.pl', write, S),
    format(S, '%% Pesos iniciales para evaluación (persistidos)~n', []),
    format(S, ':- dynamic weight/2.~n~n', []),
    forall(weight(Name,Val), format(S, 'weight(~w, ~w).~n', [Name, Val])),
    close(S).

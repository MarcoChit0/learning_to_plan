


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d367 d471 d438 )
(:init
(smaller peg1 d367)
(smaller peg1 d471)
(smaller peg1 d438)
(smaller peg2 d367)
(smaller peg2 d471)
(smaller peg2 d438)
(smaller peg3 d367)
(smaller peg3 d471)
(smaller peg3 d438)
(smaller d471 d367)
(smaller d438 d367)
(smaller d438 d471)
(clear peg2)
(clear peg3)
(clear d367)
(on d438 peg1)
(on d471 d438)
(on d367 d471)
)
(:goal
(and 
(on d438 peg3)
(on d471 d438)
(on d367 d471)
)
)
)



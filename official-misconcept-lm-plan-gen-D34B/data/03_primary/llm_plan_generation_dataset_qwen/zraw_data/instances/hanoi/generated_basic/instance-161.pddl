


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d483 d466 d61 d204 )
(:init
(smaller peg1 d483)
(smaller peg1 d466)
(smaller peg1 d61)
(smaller peg1 d204)
(smaller peg2 d483)
(smaller peg2 d466)
(smaller peg2 d61)
(smaller peg2 d204)
(smaller peg3 d483)
(smaller peg3 d466)
(smaller peg3 d61)
(smaller peg3 d204)
(smaller d466 d483)
(smaller d61 d483)
(smaller d204 d483)
(smaller d61 d466)
(smaller d204 d466)
(smaller d204 d61)
(clear peg2)
(clear peg3)
(clear d483)
(on d204 peg1)
(on d61 d204)
(on d466 d61)
(on d483 d466)
)
(:goal
(and 
(on d204 peg3)
(on d61 d204)
(on d466 d61)
(on d483 d466)
)
)
)



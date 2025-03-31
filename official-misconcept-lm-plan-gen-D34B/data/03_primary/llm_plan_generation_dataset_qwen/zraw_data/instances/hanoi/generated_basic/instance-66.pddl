


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d267 d482 )
(:init
(smaller peg1 d267)
(smaller peg1 d482)
(smaller peg2 d267)
(smaller peg2 d482)
(smaller peg3 d267)
(smaller peg3 d482)
(smaller d482 d267)
(clear peg2)
(clear peg3)
(clear d267)
(on d482 peg1)
(on d267 d482)
)
(:goal
(and 
(on d482 peg3)
(on d267 d482)
)
)
)



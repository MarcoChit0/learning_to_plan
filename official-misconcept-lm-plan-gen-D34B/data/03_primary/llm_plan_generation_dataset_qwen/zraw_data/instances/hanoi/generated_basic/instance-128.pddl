


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d197 d70 d499 )
(:init
(smaller peg1 d197)
(smaller peg1 d70)
(smaller peg1 d499)
(smaller peg2 d197)
(smaller peg2 d70)
(smaller peg2 d499)
(smaller peg3 d197)
(smaller peg3 d70)
(smaller peg3 d499)
(smaller d70 d197)
(smaller d499 d197)
(smaller d499 d70)
(clear peg2)
(clear peg3)
(clear d197)
(on d499 peg1)
(on d70 d499)
(on d197 d70)
)
(:goal
(and 
(on d499 peg3)
(on d70 d499)
(on d197 d70)
)
)
)



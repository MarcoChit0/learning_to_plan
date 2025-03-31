


(define (problem hanoi-2)
(:domain hanoi)
(:objects peg1 peg2 peg3 d457 d67 )
(:init
(smaller peg1 d457)
(smaller peg1 d67)
(smaller peg2 d457)
(smaller peg2 d67)
(smaller peg3 d457)
(smaller peg3 d67)
(smaller d67 d457)
(clear peg2)
(clear peg3)
(clear d457)
(on d67 peg1)
(on d457 d67)
)
(:goal
(and 
(on d67 peg3)
(on d457 d67)
)
)
)



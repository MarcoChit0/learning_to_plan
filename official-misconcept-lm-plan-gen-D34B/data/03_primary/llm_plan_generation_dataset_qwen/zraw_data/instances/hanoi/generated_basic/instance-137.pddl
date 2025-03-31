


(define (problem hanoi-3)
(:domain hanoi)
(:objects peg1 peg2 peg3 d96 d457 d318 )
(:init
(smaller peg1 d96)
(smaller peg1 d457)
(smaller peg1 d318)
(smaller peg2 d96)
(smaller peg2 d457)
(smaller peg2 d318)
(smaller peg3 d96)
(smaller peg3 d457)
(smaller peg3 d318)
(smaller d457 d96)
(smaller d318 d96)
(smaller d318 d457)
(clear peg2)
(clear peg3)
(clear d96)
(on d318 peg1)
(on d457 d318)
(on d96 d457)
)
(:goal
(and 
(on d318 peg3)
(on d457 d318)
(on d96 d457)
)
)
)



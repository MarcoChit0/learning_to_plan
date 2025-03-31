


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d147 d158 d499 d227 )
(:init
(smaller peg1 d147)
(smaller peg1 d158)
(smaller peg1 d499)
(smaller peg1 d227)
(smaller peg2 d147)
(smaller peg2 d158)
(smaller peg2 d499)
(smaller peg2 d227)
(smaller peg3 d147)
(smaller peg3 d158)
(smaller peg3 d499)
(smaller peg3 d227)
(smaller d158 d147)
(smaller d499 d147)
(smaller d227 d147)
(smaller d499 d158)
(smaller d227 d158)
(smaller d227 d499)
(clear peg2)
(clear peg3)
(clear d147)
(on d227 peg1)
(on d499 d227)
(on d158 d499)
(on d147 d158)
)
(:goal
(and 
(on d227 peg3)
(on d499 d227)
(on d158 d499)
(on d147 d158)
)
)
)



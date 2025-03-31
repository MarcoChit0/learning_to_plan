


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d337 d105 d212 d350 )
(:init
(smaller peg1 d337)
(smaller peg1 d105)
(smaller peg1 d212)
(smaller peg1 d350)
(smaller peg2 d337)
(smaller peg2 d105)
(smaller peg2 d212)
(smaller peg2 d350)
(smaller peg3 d337)
(smaller peg3 d105)
(smaller peg3 d212)
(smaller peg3 d350)
(smaller d105 d337)
(smaller d212 d337)
(smaller d350 d337)
(smaller d212 d105)
(smaller d350 d105)
(smaller d350 d212)
(clear peg2)
(clear peg3)
(clear d337)
(on d350 peg1)
(on d212 d350)
(on d105 d212)
(on d337 d105)
)
(:goal
(and 
(on d350 peg3)
(on d212 d350)
(on d105 d212)
(on d337 d105)
)
)
)



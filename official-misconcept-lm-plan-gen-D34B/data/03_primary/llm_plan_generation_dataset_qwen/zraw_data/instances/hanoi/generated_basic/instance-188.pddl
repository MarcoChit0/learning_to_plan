


(define (problem hanoi-4)
(:domain hanoi)
(:objects peg1 peg2 peg3 d426 d272 d404 d368 )
(:init
(smaller peg1 d426)
(smaller peg1 d272)
(smaller peg1 d404)
(smaller peg1 d368)
(smaller peg2 d426)
(smaller peg2 d272)
(smaller peg2 d404)
(smaller peg2 d368)
(smaller peg3 d426)
(smaller peg3 d272)
(smaller peg3 d404)
(smaller peg3 d368)
(smaller d272 d426)
(smaller d404 d426)
(smaller d368 d426)
(smaller d404 d272)
(smaller d368 d272)
(smaller d368 d404)
(clear peg2)
(clear peg3)
(clear d426)
(on d368 peg1)
(on d404 d368)
(on d272 d404)
(on d426 d272)
)
(:goal
(and 
(on d368 peg3)
(on d404 d368)
(on d272 d404)
(on d426 d272)
)
)
)



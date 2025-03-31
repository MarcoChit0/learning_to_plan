

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(ontable b)
(on c a)
(on d e)
(on e b)
(on f g)
(on g d)
(clear c)
)
(:goal
(and
(on a e)
(on b c)
(on d g)
(on f d)
(on g b))
)
)



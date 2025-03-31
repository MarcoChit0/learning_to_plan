

(define (problem BW-rand-6)
(:domain blocksworld-4ops)
(:objects a b c d e f )
(:init
(handempty)
(on a f)
(ontable b)
(on c e)
(on d c)
(on e b)
(on f d)
(clear a)
)
(:goal
(and
(on a e)
(on b c)
(on d b)
(on e d))
)
)



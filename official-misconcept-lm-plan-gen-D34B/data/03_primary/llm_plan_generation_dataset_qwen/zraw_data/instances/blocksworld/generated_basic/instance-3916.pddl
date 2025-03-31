

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(ontable a)
(on b a)
(on c e)
(on d b)
(on e d)
(clear c)
)
(:goal
(and
(on b d)
(on c e)
(on d a)
(on e b))
)
)





(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a e)
(ontable b)
(on c b)
(on d c)
(on e d)
(clear a)
)
(:goal
(and
(on b c)
(on c e)
(on e d))
)
)



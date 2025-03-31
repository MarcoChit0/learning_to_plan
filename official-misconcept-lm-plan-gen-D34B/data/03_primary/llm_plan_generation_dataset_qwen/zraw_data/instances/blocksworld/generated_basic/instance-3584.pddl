

(define (problem BW-rand-5)
(:domain blocksworld-4ops)
(:objects a b c d e )
(:init
(handempty)
(on a b)
(ontable b)
(on c e)
(on d a)
(on e d)
(clear c)
)
(:goal
(and
(on a c)
(on b d)
(on d e)
(on e a))
)
)



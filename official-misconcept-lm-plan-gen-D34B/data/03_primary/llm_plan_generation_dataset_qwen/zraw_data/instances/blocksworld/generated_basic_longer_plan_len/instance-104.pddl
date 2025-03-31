

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a e)
(on b c)
(ontable c)
(on d g)
(on e b)
(on f d)
(on g a)
(on h f)
(clear h)
)
(:goal
(and
(on a e)
(on b d)
(on c g)
(on e b)
(on g h)
(on h a))
)
)



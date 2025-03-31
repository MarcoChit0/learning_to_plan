

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b e)
(on c b)
(ontable d)
(on e d)
(on f c)
(on g f)
(on h a)
(clear g)
(clear h)
)
(:goal
(and
(on a h)
(on b c)
(on c g)
(on d e)
(on f b)
(on g d)
(on h f))
)
)



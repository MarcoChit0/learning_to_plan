

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a b)
(on b g)
(on c f)
(on d h)
(on e a)
(on f d)
(ontable g)
(on h e)
(clear c)
)
(:goal
(and
(on a h)
(on b g)
(on e d)
(on f b)
(on g c)
(on h f))
)
)



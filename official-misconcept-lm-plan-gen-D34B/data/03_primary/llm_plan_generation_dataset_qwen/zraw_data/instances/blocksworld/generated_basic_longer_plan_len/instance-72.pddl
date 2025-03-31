

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b g)
(on c d)
(on d a)
(on e f)
(on f b)
(on g c)
(clear e)
)
(:goal
(and
(on a f)
(on b e)
(on d c)
(on e d)
(on f g)
(on g b))
)
)



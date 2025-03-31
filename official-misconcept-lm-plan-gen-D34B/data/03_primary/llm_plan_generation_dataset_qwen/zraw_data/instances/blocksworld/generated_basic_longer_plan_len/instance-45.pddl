

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b e)
(on c f)
(on d b)
(on e c)
(on f g)
(on g a)
(clear d)
)
(:goal
(and
(on a b)
(on b f)
(on d e)
(on e a)
(on f c)
(on g d))
)
)



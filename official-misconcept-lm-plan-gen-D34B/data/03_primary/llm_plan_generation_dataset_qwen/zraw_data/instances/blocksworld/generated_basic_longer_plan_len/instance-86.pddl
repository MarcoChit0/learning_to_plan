

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a b)
(on b c)
(ontable c)
(on d f)
(on e a)
(on f g)
(on g e)
(clear d)
)
(:goal
(and
(on a f)
(on b g)
(on d c)
(on e d)
(on f b)
(on g e))
)
)



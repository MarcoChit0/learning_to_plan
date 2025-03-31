

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b g)
(ontable c)
(on d b)
(on e a)
(on f d)
(on g e)
(clear f)
)
(:goal
(and
(on b d)
(on c b)
(on d a)
(on e f)
(on f c)
(on g e))
)
)



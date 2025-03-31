

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b a)
(on c b)
(on d e)
(on e c)
(on f g)
(on g d)
(clear f)
)
(:goal
(and
(on a c)
(on b d)
(on d f)
(on e a)
(on f e))
)
)





(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b d)
(on c f)
(on d a)
(on e b)
(on f g)
(on g e)
(clear c)
)
(:goal
(and
(on a c)
(on c e)
(on d f)
(on f a)
(on g d))
)
)





(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a e)
(ontable b)
(on c a)
(ontable d)
(on e b)
(on f g)
(on g c)
(clear d)
(clear f)
)
(:goal
(and
(on a e)
(on c f)
(on d b)
(on e g)
(on f a)
(on g d))
)
)



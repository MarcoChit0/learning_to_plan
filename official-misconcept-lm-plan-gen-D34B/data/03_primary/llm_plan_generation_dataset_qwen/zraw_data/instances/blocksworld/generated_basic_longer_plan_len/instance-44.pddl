

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b f)
(on c e)
(on d b)
(on e g)
(on f a)
(on g d)
(clear c)
)
(:goal
(and
(on a f)
(on b a)
(on c e)
(on d b)
(on e g))
)
)



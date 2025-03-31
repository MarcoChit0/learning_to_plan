

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a d)
(on b a)
(on c b)
(on d f)
(on e g)
(ontable f)
(ontable g)
(clear c)
(clear e)
)
(:goal
(and
(on a b)
(on b c)
(on c e)
(on e f)
(on f g)
(on g d))
)
)



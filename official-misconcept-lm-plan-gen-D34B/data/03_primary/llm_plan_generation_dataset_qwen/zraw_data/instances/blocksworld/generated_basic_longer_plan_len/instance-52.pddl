

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a c)
(on b e)
(on c b)
(on d a)
(on e g)
(ontable f)
(ontable g)
(clear d)
(clear f)
)
(:goal
(and
(on a e)
(on b d)
(on c b)
(on d a)
(on e f)
(on f g))
)
)



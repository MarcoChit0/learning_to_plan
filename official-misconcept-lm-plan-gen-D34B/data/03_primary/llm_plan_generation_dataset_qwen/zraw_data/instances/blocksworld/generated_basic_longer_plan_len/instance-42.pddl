

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b a)
(on c b)
(ontable d)
(on e g)
(on f e)
(on g d)
(clear c)
(clear f)
)
(:goal
(and
(on a g)
(on b c)
(on c a)
(on e d)
(on f b)
(on g e))
)
)



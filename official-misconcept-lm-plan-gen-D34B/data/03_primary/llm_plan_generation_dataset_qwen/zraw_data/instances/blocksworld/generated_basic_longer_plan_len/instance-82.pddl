

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(ontable a)
(on b a)
(on c b)
(ontable d)
(on e d)
(on f c)
(on g f)
(clear e)
(clear g)
)
(:goal
(and
(on a f)
(on c e)
(on d b)
(on e g)
(on f d)
(on g a))
)
)



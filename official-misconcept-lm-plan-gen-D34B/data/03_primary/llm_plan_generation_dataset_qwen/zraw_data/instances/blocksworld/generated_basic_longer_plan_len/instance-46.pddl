

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b c)
(ontable c)
(on d a)
(on e g)
(on f b)
(on g d)
(clear e)
)
(:goal
(and
(on b c)
(on c g)
(on d a)
(on e d)
(on f b)
(on g e))
)
)



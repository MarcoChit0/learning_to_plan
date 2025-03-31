

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b d)
(on c e)
(on d c)
(on e a)
(on f g)
(ontable g)
(clear b)
)
(:goal
(and
(on a f)
(on c d)
(on d e)
(on e g)
(on g a))
)
)



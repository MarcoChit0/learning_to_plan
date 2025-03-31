

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a f)
(on b e)
(on c a)
(on d c)
(on e g)
(on f b)
(ontable g)
(clear d)
)
(:goal
(and
(on a b)
(on c f)
(on d g)
(on e d)
(on f a))
)
)



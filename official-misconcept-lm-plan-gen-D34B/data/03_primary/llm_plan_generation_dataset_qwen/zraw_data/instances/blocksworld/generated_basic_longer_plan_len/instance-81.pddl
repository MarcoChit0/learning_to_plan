

(define (problem BW-rand-7)
(:domain blocksworld-4ops)
(:objects a b c d e f g )
(:init
(handempty)
(on a g)
(on b a)
(on c f)
(on d c)
(on e b)
(on f e)
(ontable g)
(clear d)
)
(:goal
(and
(on a c)
(on b a)
(on d e)
(on e g)
(on f b)
(on g f))
)
)



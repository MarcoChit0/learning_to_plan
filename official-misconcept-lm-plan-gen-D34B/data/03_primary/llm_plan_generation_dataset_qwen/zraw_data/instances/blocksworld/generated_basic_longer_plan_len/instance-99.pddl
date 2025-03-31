

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(ontable b)
(on c h)
(on d c)
(on e g)
(on f b)
(on g a)
(on h f)
(clear d)
(clear e)
)
(:goal
(and
(on a e)
(on b c)
(on c h)
(on d g)
(on f d)
(on g b)
(on h a))
)
)



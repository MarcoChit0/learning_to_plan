

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b h)
(on c f)
(on d c)
(on e g)
(on f e)
(on g a)
(ontable h)
(clear b)
(clear d)
)
(:goal
(and
(on a e)
(on b d)
(on c f)
(on d a)
(on e h)
(on g b)
(on h c))
)
)



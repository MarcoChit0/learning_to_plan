

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(ontable a)
(on b d)
(on c h)
(on d a)
(on e f)
(ontable f)
(on g e)
(on h g)
(clear b)
(clear c)
)
(:goal
(and
(on a g)
(on b h)
(on d b)
(on e a)
(on f c)
(on g f)
(on h e))
)
)



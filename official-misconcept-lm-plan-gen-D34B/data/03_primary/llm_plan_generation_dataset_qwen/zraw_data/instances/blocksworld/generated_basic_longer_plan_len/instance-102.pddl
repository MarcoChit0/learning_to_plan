

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a f)
(on b d)
(on c g)
(ontable d)
(on e h)
(ontable f)
(on g a)
(on h c)
(clear b)
(clear e)
)
(:goal
(and
(on a b)
(on b f)
(on c d)
(on d h)
(on e a)
(on f c)
(on g e))
)
)



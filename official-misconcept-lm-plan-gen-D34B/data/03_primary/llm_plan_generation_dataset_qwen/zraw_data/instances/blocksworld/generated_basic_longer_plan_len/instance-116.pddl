

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a d)
(on b h)
(ontable c)
(on d c)
(on e a)
(on f g)
(on g e)
(ontable h)
(clear b)
(clear f)
)
(:goal
(and
(on a e)
(on c g)
(on d h)
(on e d)
(on f c)
(on g a)
(on h b))
)
)



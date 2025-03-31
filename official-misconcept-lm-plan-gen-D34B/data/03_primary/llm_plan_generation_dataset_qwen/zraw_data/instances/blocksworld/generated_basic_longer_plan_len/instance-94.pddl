

(define (problem BW-rand-8)
(:domain blocksworld-4ops)
(:objects a b c d e f g h )
(:init
(handempty)
(on a c)
(on b a)
(ontable c)
(on d e)
(on e h)
(on f d)
(on g f)
(on h b)
(clear g)
)
(:goal
(and
(on a f)
(on b c)
(on c g)
(on d a)
(on f b)
(on g e))
)
)



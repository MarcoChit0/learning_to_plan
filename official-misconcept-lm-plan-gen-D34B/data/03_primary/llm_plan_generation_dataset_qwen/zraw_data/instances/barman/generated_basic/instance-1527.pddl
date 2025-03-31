(define (problem prob)
 (:domain barman)
 (:objects 
      shaker156 - shaker
      left right - hand
      shot489 shot406 - shot
      ingredient324 ingredient108 ingredient374 ingredient331 - ingredient
      cocktail73 - cocktail
      dispenser141 dispenser38 dispenser286 dispenser281 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker156)
  (ontable shot489)
  (ontable shot406)
  (dispenses dispenser141 ingredient324)
  (dispenses dispenser38 ingredient108)
  (dispenses dispenser286 ingredient374)
  (dispenses dispenser281 ingredient331)
  (clean shaker156)
  (clean shot489)
  (clean shot406)
  (empty shaker156)
  (empty shot489)
  (empty shot406)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker156 l0)
  (shaker-level shaker156 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail73 ingredient108)
  (cocktail-part2 cocktail73 ingredient324)
)
 (:goal
  (and
      (contains shot489 cocktail73)
)))

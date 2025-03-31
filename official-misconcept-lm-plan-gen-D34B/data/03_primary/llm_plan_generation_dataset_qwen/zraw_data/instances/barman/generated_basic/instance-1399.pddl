(define (problem prob)
 (:domain barman)
 (:objects 
      shaker202 - shaker
      left right - hand
      shot83 - shot
      ingredient185 ingredient176 ingredient25 ingredient210 - ingredient
      cocktail451 - cocktail
      dispenser370 dispenser388 dispenser209 dispenser489 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker202)
  (ontable shot83)
  (dispenses dispenser370 ingredient185)
  (dispenses dispenser388 ingredient176)
  (dispenses dispenser209 ingredient25)
  (dispenses dispenser489 ingredient210)
  (clean shaker202)
  (clean shot83)
  (empty shaker202)
  (empty shot83)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker202 l0)
  (shaker-level shaker202 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail451 ingredient176)
  (cocktail-part2 cocktail451 ingredient210)
)
 (:goal
  (and
      (contains shot83 cocktail451)
)))

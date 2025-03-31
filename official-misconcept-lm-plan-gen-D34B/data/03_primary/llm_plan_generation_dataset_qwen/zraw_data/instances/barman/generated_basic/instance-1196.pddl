(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot397 shot57 - shot
      ingredient142 ingredient446 - ingredient
      cocktail450 - cocktail
      dispenser393 dispenser371 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot397)
  (ontable shot57)
  (dispenses dispenser393 ingredient142)
  (dispenses dispenser371 ingredient446)
  (clean shaker281)
  (clean shot397)
  (clean shot57)
  (empty shaker281)
  (empty shot397)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail450 ingredient142)
  (cocktail-part2 cocktail450 ingredient446)
)
 (:goal
  (and
      (contains shot397 cocktail450)
)))

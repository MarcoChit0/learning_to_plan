(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot311 - shot
      ingredient297 ingredient435 ingredient299 ingredient422 - ingredient
      cocktail30 - cocktail
      dispenser358 dispenser445 dispenser208 dispenser282 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot311)
  (dispenses dispenser358 ingredient297)
  (dispenses dispenser445 ingredient435)
  (dispenses dispenser208 ingredient299)
  (dispenses dispenser282 ingredient422)
  (clean shaker472)
  (clean shot311)
  (empty shaker472)
  (empty shot311)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail30 ingredient422)
  (cocktail-part2 cocktail30 ingredient299)
)
 (:goal
  (and
      (contains shot311 cocktail30)
)))

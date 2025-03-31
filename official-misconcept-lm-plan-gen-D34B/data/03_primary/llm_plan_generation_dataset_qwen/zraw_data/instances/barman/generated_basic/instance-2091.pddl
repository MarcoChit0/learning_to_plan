(define (problem prob)
 (:domain barman)
 (:objects 
      shaker89 - shaker
      left right - hand
      shot419 shot211 - shot
      ingredient490 ingredient180 - ingredient
      cocktail72 - cocktail
      dispenser341 dispenser21 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker89)
  (ontable shot419)
  (ontable shot211)
  (dispenses dispenser341 ingredient490)
  (dispenses dispenser21 ingredient180)
  (clean shaker89)
  (clean shot419)
  (clean shot211)
  (empty shaker89)
  (empty shot419)
  (empty shot211)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker89 l0)
  (shaker-level shaker89 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail72 ingredient180)
  (cocktail-part2 cocktail72 ingredient490)
)
 (:goal
  (and
      (contains shot419 cocktail72)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker344 - shaker
      left right - hand
      shot192 shot114 shot419 - shot
      ingredient184 ingredient281 ingredient40 - ingredient
      cocktail1 - cocktail
      dispenser421 dispenser491 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker344)
  (ontable shot192)
  (ontable shot114)
  (ontable shot419)
  (dispenses dispenser421 ingredient184)
  (dispenses dispenser491 ingredient281)
  (dispenses dispenser393 ingredient40)
  (clean shaker344)
  (clean shot192)
  (clean shot114)
  (clean shot419)
  (empty shaker344)
  (empty shot192)
  (empty shot114)
  (empty shot419)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker344 l0)
  (shaker-level shaker344 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient40)
  (cocktail-part2 cocktail1 ingredient281)
)
 (:goal
  (and
      (contains shot192 cocktail1)
      (contains shot114 cocktail1)
)))

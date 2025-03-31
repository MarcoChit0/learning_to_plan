(define (problem prob)
 (:domain barman)
 (:objects 
      shaker371 - shaker
      left right - hand
      shot99 shot498 - shot
      ingredient487 ingredient430 - ingredient
      cocktail355 - cocktail
      dispenser427 dispenser59 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker371)
  (ontable shot99)
  (ontable shot498)
  (dispenses dispenser427 ingredient487)
  (dispenses dispenser59 ingredient430)
  (clean shaker371)
  (clean shot99)
  (clean shot498)
  (empty shaker371)
  (empty shot99)
  (empty shot498)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker371 l0)
  (shaker-level shaker371 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail355 ingredient487)
  (cocktail-part2 cocktail355 ingredient430)
)
 (:goal
  (and
      (contains shot99 cocktail355)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot457 shot329 shot244 - shot
      ingredient27 ingredient79 ingredient346 ingredient495 - ingredient
      cocktail62 - cocktail
      dispenser217 dispenser191 dispenser82 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot457)
  (ontable shot329)
  (ontable shot244)
  (dispenses dispenser217 ingredient27)
  (dispenses dispenser191 ingredient79)
  (dispenses dispenser82 ingredient346)
  (dispenses dispenser45 ingredient495)
  (clean shaker484)
  (clean shot457)
  (clean shot329)
  (clean shot244)
  (empty shaker484)
  (empty shot457)
  (empty shot329)
  (empty shot244)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient79)
  (cocktail-part2 cocktail62 ingredient495)
)
 (:goal
  (and
      (contains shot457 cocktail62)
      (contains shot329 cocktail62)
)))

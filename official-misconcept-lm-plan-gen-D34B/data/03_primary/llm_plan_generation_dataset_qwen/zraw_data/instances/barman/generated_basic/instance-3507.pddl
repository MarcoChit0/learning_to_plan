(define (problem prob)
 (:domain barman)
 (:objects 
      shaker366 - shaker
      left right - hand
      shot110 shot94 - shot
      ingredient294 ingredient450 ingredient91 ingredient393 - ingredient
      cocktail321 - cocktail
      dispenser7 dispenser294 dispenser338 dispenser418 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker366)
  (ontable shot110)
  (ontable shot94)
  (dispenses dispenser7 ingredient294)
  (dispenses dispenser294 ingredient450)
  (dispenses dispenser338 ingredient91)
  (dispenses dispenser418 ingredient393)
  (clean shaker366)
  (clean shot110)
  (clean shot94)
  (empty shaker366)
  (empty shot110)
  (empty shot94)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker366 l0)
  (shaker-level shaker366 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail321 ingredient450)
  (cocktail-part2 cocktail321 ingredient393)
)
 (:goal
  (and
      (contains shot110 cocktail321)
)))

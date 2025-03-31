(define (problem prob)
 (:domain barman)
 (:objects 
      shaker465 - shaker
      left right - hand
      shot90 shot452 - shot
      ingredient487 ingredient133 ingredient98 - ingredient
      cocktail1 - cocktail
      dispenser154 dispenser419 dispenser330 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker465)
  (ontable shot90)
  (ontable shot452)
  (dispenses dispenser154 ingredient487)
  (dispenses dispenser419 ingredient133)
  (dispenses dispenser330 ingredient98)
  (clean shaker465)
  (clean shot90)
  (clean shot452)
  (empty shaker465)
  (empty shot90)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker465 l0)
  (shaker-level shaker465 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient98)
  (cocktail-part2 cocktail1 ingredient487)
)
 (:goal
  (and
      (contains shot90 cocktail1)
)))

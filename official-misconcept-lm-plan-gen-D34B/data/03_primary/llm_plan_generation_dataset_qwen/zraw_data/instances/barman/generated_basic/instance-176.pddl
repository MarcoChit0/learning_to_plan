(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot382 shot116 shot336 - shot
      ingredient497 ingredient360 ingredient348 ingredient23 - ingredient
      cocktail1 - cocktail
      dispenser421 dispenser276 dispenser214 dispenser201 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot382)
  (ontable shot116)
  (ontable shot336)
  (dispenses dispenser421 ingredient497)
  (dispenses dispenser276 ingredient360)
  (dispenses dispenser214 ingredient348)
  (dispenses dispenser201 ingredient23)
  (clean shaker17)
  (clean shot382)
  (clean shot116)
  (clean shot336)
  (empty shaker17)
  (empty shot382)
  (empty shot116)
  (empty shot336)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient360)
  (cocktail-part2 cocktail1 ingredient23)
)
 (:goal
  (and
      (contains shot382 cocktail1)
      (contains shot116 cocktail1)
)))

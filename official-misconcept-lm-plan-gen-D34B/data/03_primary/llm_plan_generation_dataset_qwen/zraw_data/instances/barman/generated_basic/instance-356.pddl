(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot311 shot65 - shot
      ingredient91 ingredient115 ingredient200 ingredient65 - ingredient
      cocktail1 - cocktail
      dispenser91 dispenser114 dispenser325 dispenser88 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot311)
  (ontable shot65)
  (dispenses dispenser91 ingredient91)
  (dispenses dispenser114 ingredient115)
  (dispenses dispenser325 ingredient200)
  (dispenses dispenser88 ingredient65)
  (clean shaker426)
  (clean shot311)
  (clean shot65)
  (empty shaker426)
  (empty shot311)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient200)
  (cocktail-part2 cocktail1 ingredient65)
)
 (:goal
  (and
      (contains shot311 cocktail1)
)))

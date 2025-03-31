(define (problem prob)
 (:domain barman)
 (:objects 
      shaker189 - shaker
      left right - hand
      shot472 shot72 shot169 - shot
      ingredient63 ingredient418 ingredient179 ingredient120 - ingredient
      cocktail1 - cocktail
      dispenser113 dispenser297 dispenser201 dispenser54 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker189)
  (ontable shot472)
  (ontable shot72)
  (ontable shot169)
  (dispenses dispenser113 ingredient63)
  (dispenses dispenser297 ingredient418)
  (dispenses dispenser201 ingredient179)
  (dispenses dispenser54 ingredient120)
  (clean shaker189)
  (clean shot472)
  (clean shot72)
  (clean shot169)
  (empty shaker189)
  (empty shot472)
  (empty shot72)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker189 l0)
  (shaker-level shaker189 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient418)
  (cocktail-part2 cocktail1 ingredient179)
)
 (:goal
  (and
      (contains shot472 cocktail1)
      (contains shot72 ingredient63)
)))

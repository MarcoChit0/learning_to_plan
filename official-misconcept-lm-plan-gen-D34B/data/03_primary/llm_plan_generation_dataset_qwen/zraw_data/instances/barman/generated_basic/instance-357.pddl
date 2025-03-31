(define (problem prob)
 (:domain barman)
 (:objects 
      shaker470 - shaker
      left right - hand
      shot388 shot81 - shot
      ingredient373 ingredient347 ingredient175 ingredient495 - ingredient
      cocktail1 - cocktail
      dispenser274 dispenser394 dispenser258 dispenser80 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker470)
  (ontable shot388)
  (ontable shot81)
  (dispenses dispenser274 ingredient373)
  (dispenses dispenser394 ingredient347)
  (dispenses dispenser258 ingredient175)
  (dispenses dispenser80 ingredient495)
  (clean shaker470)
  (clean shot388)
  (clean shot81)
  (empty shaker470)
  (empty shot388)
  (empty shot81)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker470 l0)
  (shaker-level shaker470 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient495)
  (cocktail-part2 cocktail1 ingredient175)
)
 (:goal
  (and
      (contains shot388 cocktail1)
)))

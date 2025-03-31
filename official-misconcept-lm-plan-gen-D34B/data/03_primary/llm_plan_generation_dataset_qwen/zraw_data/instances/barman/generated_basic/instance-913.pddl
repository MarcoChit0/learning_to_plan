(define (problem prob)
 (:domain barman)
 (:objects 
      shaker46 - shaker
      left right - hand
      shot114 shot250 - shot
      ingredient100 ingredient124 ingredient390 - ingredient
      cocktail1 - cocktail
      dispenser396 dispenser231 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker46)
  (ontable shot114)
  (ontable shot250)
  (dispenses dispenser396 ingredient100)
  (dispenses dispenser231 ingredient124)
  (dispenses dispenser362 ingredient390)
  (clean shaker46)
  (clean shot114)
  (clean shot250)
  (empty shaker46)
  (empty shot114)
  (empty shot250)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker46 l0)
  (shaker-level shaker46 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient100)
  (cocktail-part2 cocktail1 ingredient390)
)
 (:goal
  (and
      (contains shot114 cocktail1)
)))

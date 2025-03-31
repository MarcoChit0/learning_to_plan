(define (problem prob)
 (:domain barman)
 (:objects 
      shaker62 - shaker
      left right - hand
      shot325 shot7 - shot
      ingredient62 ingredient365 - ingredient
      cocktail1 - cocktail
      dispenser118 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker62)
  (ontable shot325)
  (ontable shot7)
  (dispenses dispenser118 ingredient62)
  (dispenses dispenser46 ingredient365)
  (clean shaker62)
  (clean shot325)
  (clean shot7)
  (empty shaker62)
  (empty shot325)
  (empty shot7)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker62 l0)
  (shaker-level shaker62 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient62)
  (cocktail-part2 cocktail1 ingredient365)
)
 (:goal
  (and
      (contains shot325 cocktail1)
)))

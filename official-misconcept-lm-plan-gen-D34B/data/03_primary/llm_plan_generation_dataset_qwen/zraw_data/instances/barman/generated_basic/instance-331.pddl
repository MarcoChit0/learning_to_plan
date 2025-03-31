(define (problem prob)
 (:domain barman)
 (:objects 
      shaker324 - shaker
      left right - hand
      shot130 - shot
      ingredient6 ingredient78 - ingredient
      cocktail1 - cocktail
      dispenser434 dispenser219 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker324)
  (ontable shot130)
  (dispenses dispenser434 ingredient6)
  (dispenses dispenser219 ingredient78)
  (clean shaker324)
  (clean shot130)
  (empty shaker324)
  (empty shot130)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker324 l0)
  (shaker-level shaker324 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient78)
  (cocktail-part2 cocktail1 ingredient6)
)
 (:goal
  (and
      (contains shot130 cocktail1)
)))

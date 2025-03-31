(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot92 shot387 shot430 - shot
      ingredient266 ingredient376 ingredient1 ingredient258 - ingredient
      cocktail44 - cocktail
      dispenser92 dispenser6 dispenser34 dispenser440 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot92)
  (ontable shot387)
  (ontable shot430)
  (dispenses dispenser92 ingredient266)
  (dispenses dispenser6 ingredient376)
  (dispenses dispenser34 ingredient1)
  (dispenses dispenser440 ingredient258)
  (clean shaker234)
  (clean shot92)
  (clean shot387)
  (clean shot430)
  (empty shaker234)
  (empty shot92)
  (empty shot387)
  (empty shot430)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient1)
  (cocktail-part2 cocktail44 ingredient258)
)
 (:goal
  (and
      (contains shot92 cocktail44)
      (contains shot387 cocktail44)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker61 - shaker
      left right - hand
      shot99 shot130 shot96 shot72 - shot
      ingredient241 ingredient376 ingredient269 - ingredient
      cocktail339 - cocktail
      dispenser367 dispenser293 dispenser214 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker61)
  (ontable shot99)
  (ontable shot130)
  (ontable shot96)
  (ontable shot72)
  (dispenses dispenser367 ingredient241)
  (dispenses dispenser293 ingredient376)
  (dispenses dispenser214 ingredient269)
  (clean shaker61)
  (clean shot99)
  (clean shot130)
  (clean shot96)
  (clean shot72)
  (empty shaker61)
  (empty shot99)
  (empty shot130)
  (empty shot96)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker61 l0)
  (shaker-level shaker61 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail339 ingredient241)
  (cocktail-part2 cocktail339 ingredient376)
)
 (:goal
  (and
      (contains shot99 cocktail339)
      (contains shot130 ingredient376)
      (contains shot96 cocktail339)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot462 shot38 - shot
      ingredient283 ingredient431 - ingredient
      cocktail50 - cocktail
      dispenser286 dispenser43 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot462)
  (ontable shot38)
  (dispenses dispenser286 ingredient283)
  (dispenses dispenser43 ingredient431)
  (clean shaker472)
  (clean shot462)
  (clean shot38)
  (empty shaker472)
  (empty shot462)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail50 ingredient431)
  (cocktail-part2 cocktail50 ingredient283)
)
 (:goal
  (and
      (contains shot462 cocktail50)
)))

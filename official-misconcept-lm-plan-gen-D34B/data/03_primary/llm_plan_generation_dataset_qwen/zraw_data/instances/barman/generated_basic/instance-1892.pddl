(define (problem prob)
 (:domain barman)
 (:objects 
      shaker240 - shaker
      left right - hand
      shot455 shot328 - shot
      ingredient78 ingredient431 ingredient13 - ingredient
      cocktail312 - cocktail
      dispenser409 dispenser479 dispenser27 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker240)
  (ontable shot455)
  (ontable shot328)
  (dispenses dispenser409 ingredient78)
  (dispenses dispenser479 ingredient431)
  (dispenses dispenser27 ingredient13)
  (clean shaker240)
  (clean shot455)
  (clean shot328)
  (empty shaker240)
  (empty shot455)
  (empty shot328)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker240 l0)
  (shaker-level shaker240 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail312 ingredient78)
  (cocktail-part2 cocktail312 ingredient13)
)
 (:goal
  (and
      (contains shot455 cocktail312)
)))

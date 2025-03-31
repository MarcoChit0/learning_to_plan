(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot131 shot67 - shot
      ingredient395 ingredient415 ingredient481 - ingredient
      cocktail262 - cocktail
      dispenser285 dispenser118 dispenser369 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot131)
  (ontable shot67)
  (dispenses dispenser285 ingredient395)
  (dispenses dispenser118 ingredient415)
  (dispenses dispenser369 ingredient481)
  (clean shaker498)
  (clean shot131)
  (clean shot67)
  (empty shaker498)
  (empty shot131)
  (empty shot67)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail262 ingredient415)
  (cocktail-part2 cocktail262 ingredient395)
)
 (:goal
  (and
      (contains shot131 cocktail262)
)))

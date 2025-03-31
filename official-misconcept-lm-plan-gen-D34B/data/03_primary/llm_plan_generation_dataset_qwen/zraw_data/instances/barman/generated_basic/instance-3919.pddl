(define (problem prob)
 (:domain barman)
 (:objects 
      shaker469 - shaker
      left right - hand
      shot347 - shot
      ingredient474 ingredient96 ingredient368 ingredient331 - ingredient
      cocktail472 - cocktail
      dispenser57 dispenser24 dispenser54 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker469)
  (ontable shot347)
  (dispenses dispenser57 ingredient474)
  (dispenses dispenser24 ingredient96)
  (dispenses dispenser54 ingredient368)
  (dispenses dispenser306 ingredient331)
  (clean shaker469)
  (clean shot347)
  (empty shaker469)
  (empty shot347)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker469 l0)
  (shaker-level shaker469 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail472 ingredient474)
  (cocktail-part2 cocktail472 ingredient331)
)
 (:goal
  (and
      (contains shot347 cocktail472)
)))

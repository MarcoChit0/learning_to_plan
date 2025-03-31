(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot443 - shot
      ingredient480 ingredient308 - ingredient
      cocktail314 - cocktail
      dispenser320 dispenser479 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot443)
  (dispenses dispenser320 ingredient480)
  (dispenses dispenser479 ingredient308)
  (clean shaker358)
  (clean shot443)
  (empty shaker358)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail314 ingredient308)
  (cocktail-part2 cocktail314 ingredient480)
)
 (:goal
  (and
      (contains shot443 cocktail314)
)))

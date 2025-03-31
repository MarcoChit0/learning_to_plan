(define (problem prob)
 (:domain barman)
 (:objects 
      shaker484 - shaker
      left right - hand
      shot111 - shot
      ingredient169 ingredient272 ingredient434 - ingredient
      cocktail467 - cocktail
      dispenser325 dispenser476 dispenser129 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker484)
  (ontable shot111)
  (dispenses dispenser325 ingredient169)
  (dispenses dispenser476 ingredient272)
  (dispenses dispenser129 ingredient434)
  (clean shaker484)
  (clean shot111)
  (empty shaker484)
  (empty shot111)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker484 l0)
  (shaker-level shaker484 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail467 ingredient272)
  (cocktail-part2 cocktail467 ingredient434)
)
 (:goal
  (and
      (contains shot111 cocktail467)
)))

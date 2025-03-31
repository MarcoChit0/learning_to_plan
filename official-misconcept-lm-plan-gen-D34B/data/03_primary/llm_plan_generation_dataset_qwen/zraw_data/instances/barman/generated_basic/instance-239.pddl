(define (problem prob)
 (:domain barman)
 (:objects 
      shaker78 - shaker
      left right - hand
      shot53 - shot
      ingredient483 ingredient118 - ingredient
      cocktail1 - cocktail
      dispenser306 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker78)
  (ontable shot53)
  (dispenses dispenser306 ingredient483)
  (dispenses dispenser90 ingredient118)
  (clean shaker78)
  (clean shot53)
  (empty shaker78)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker78 l0)
  (shaker-level shaker78 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient118)
  (cocktail-part2 cocktail1 ingredient483)
)
 (:goal
  (and
      (contains shot53 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker86 - shaker
      left right - hand
      shot131 - shot
      ingredient481 ingredient146 ingredient111 - ingredient
      cocktail135 - cocktail
      dispenser118 dispenser234 dispenser240 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker86)
  (ontable shot131)
  (dispenses dispenser118 ingredient481)
  (dispenses dispenser234 ingredient146)
  (dispenses dispenser240 ingredient111)
  (clean shaker86)
  (clean shot131)
  (empty shaker86)
  (empty shot131)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker86 l0)
  (shaker-level shaker86 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail135 ingredient146)
  (cocktail-part2 cocktail135 ingredient481)
)
 (:goal
  (and
      (contains shot131 cocktail135)
)))

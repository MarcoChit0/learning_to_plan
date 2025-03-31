(define (problem prob)
 (:domain barman)
 (:objects 
      shaker139 - shaker
      left right - hand
      shot72 - shot
      ingredient382 ingredient437 ingredient222 ingredient83 - ingredient
      cocktail476 - cocktail
      dispenser91 dispenser185 dispenser25 dispenser252 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker139)
  (ontable shot72)
  (dispenses dispenser91 ingredient382)
  (dispenses dispenser185 ingredient437)
  (dispenses dispenser25 ingredient222)
  (dispenses dispenser252 ingredient83)
  (clean shaker139)
  (clean shot72)
  (empty shaker139)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker139 l0)
  (shaker-level shaker139 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail476 ingredient382)
  (cocktail-part2 cocktail476 ingredient437)
)
 (:goal
  (and
      (contains shot72 cocktail476)
)))

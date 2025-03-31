(define (problem prob)
 (:domain barman)
 (:objects 
      shaker38 - shaker
      left right - hand
      shot429 - shot
      ingredient349 ingredient443 ingredient99 - ingredient
      cocktail136 - cocktail
      dispenser494 dispenser124 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker38)
  (ontable shot429)
  (dispenses dispenser494 ingredient349)
  (dispenses dispenser124 ingredient443)
  (dispenses dispenser24 ingredient99)
  (clean shaker38)
  (clean shot429)
  (empty shaker38)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker38 l0)
  (shaker-level shaker38 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail136 ingredient443)
  (cocktail-part2 cocktail136 ingredient99)
)
 (:goal
  (and
      (contains shot429 cocktail136)
)))

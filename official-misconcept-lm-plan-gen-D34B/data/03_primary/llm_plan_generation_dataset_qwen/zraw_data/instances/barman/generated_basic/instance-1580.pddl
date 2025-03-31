(define (problem prob)
 (:domain barman)
 (:objects 
      shaker130 - shaker
      left right - hand
      shot184 shot53 - shot
      ingredient236 ingredient482 - ingredient
      cocktail266 - cocktail
      dispenser106 dispenser119 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker130)
  (ontable shot184)
  (ontable shot53)
  (dispenses dispenser106 ingredient236)
  (dispenses dispenser119 ingredient482)
  (clean shaker130)
  (clean shot184)
  (clean shot53)
  (empty shaker130)
  (empty shot184)
  (empty shot53)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker130 l0)
  (shaker-level shaker130 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail266 ingredient482)
  (cocktail-part2 cocktail266 ingredient236)
)
 (:goal
  (and
      (contains shot184 cocktail266)
)))

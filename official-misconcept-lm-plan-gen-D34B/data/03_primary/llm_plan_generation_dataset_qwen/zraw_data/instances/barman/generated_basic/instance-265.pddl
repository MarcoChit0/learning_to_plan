(define (problem prob)
 (:domain barman)
 (:objects 
      shaker262 - shaker
      left right - hand
      shot237 shot128 shot465 - shot
      ingredient288 ingredient210 ingredient481 - ingredient
      cocktail1 - cocktail
      dispenser191 dispenser493 dispenser424 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker262)
  (ontable shot237)
  (ontable shot128)
  (ontable shot465)
  (dispenses dispenser191 ingredient288)
  (dispenses dispenser493 ingredient210)
  (dispenses dispenser424 ingredient481)
  (clean shaker262)
  (clean shot237)
  (clean shot128)
  (clean shot465)
  (empty shaker262)
  (empty shot237)
  (empty shot128)
  (empty shot465)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker262 l0)
  (shaker-level shaker262 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient210)
  (cocktail-part2 cocktail1 ingredient288)
)
 (:goal
  (and
      (contains shot237 cocktail1)
      (contains shot128 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker138 - shaker
      left right - hand
      shot192 - shot
      ingredient332 ingredient243 - ingredient
      cocktail1 - cocktail
      dispenser122 dispenser293 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker138)
  (ontable shot192)
  (dispenses dispenser122 ingredient332)
  (dispenses dispenser293 ingredient243)
  (clean shaker138)
  (clean shot192)
  (empty shaker138)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker138 l0)
  (shaker-level shaker138 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient243)
  (cocktail-part2 cocktail1 ingredient332)
)
 (:goal
  (and
      (contains shot192 cocktail1)
)))

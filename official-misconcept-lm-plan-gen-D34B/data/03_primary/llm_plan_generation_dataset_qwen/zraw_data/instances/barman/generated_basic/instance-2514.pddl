(define (problem prob)
 (:domain barman)
 (:objects 
      shaker283 - shaker
      left right - hand
      shot200 - shot
      ingredient41 ingredient96 - ingredient
      cocktail183 - cocktail
      dispenser407 dispenser458 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker283)
  (ontable shot200)
  (dispenses dispenser407 ingredient41)
  (dispenses dispenser458 ingredient96)
  (clean shaker283)
  (clean shot200)
  (empty shaker283)
  (empty shot200)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker283 l0)
  (shaker-level shaker283 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail183 ingredient96)
  (cocktail-part2 cocktail183 ingredient41)
)
 (:goal
  (and
      (contains shot200 cocktail183)
)))

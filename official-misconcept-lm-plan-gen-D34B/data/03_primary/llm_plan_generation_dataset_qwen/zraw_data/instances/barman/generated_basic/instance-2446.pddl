(define (problem prob)
 (:domain barman)
 (:objects 
      shaker284 - shaker
      left right - hand
      shot397 shot287 shot475 - shot
      ingredient173 ingredient432 ingredient214 ingredient385 - ingredient
      cocktail471 - cocktail
      dispenser431 dispenser433 dispenser151 dispenser432 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker284)
  (ontable shot397)
  (ontable shot287)
  (ontable shot475)
  (dispenses dispenser431 ingredient173)
  (dispenses dispenser433 ingredient432)
  (dispenses dispenser151 ingredient214)
  (dispenses dispenser432 ingredient385)
  (clean shaker284)
  (clean shot397)
  (clean shot287)
  (clean shot475)
  (empty shaker284)
  (empty shot397)
  (empty shot287)
  (empty shot475)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker284 l0)
  (shaker-level shaker284 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail471 ingredient432)
  (cocktail-part2 cocktail471 ingredient173)
)
 (:goal
  (and
      (contains shot397 cocktail471)
      (contains shot287 cocktail471)
)))

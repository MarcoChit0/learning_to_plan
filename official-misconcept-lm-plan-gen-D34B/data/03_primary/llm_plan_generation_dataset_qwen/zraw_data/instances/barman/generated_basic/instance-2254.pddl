(define (problem prob)
 (:domain barman)
 (:objects 
      shaker279 - shaker
      left right - hand
      shot38 shot249 shot471 - shot
      ingredient379 ingredient128 ingredient295 ingredient309 - ingredient
      cocktail56 - cocktail
      dispenser419 dispenser193 dispenser324 dispenser116 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker279)
  (ontable shot38)
  (ontable shot249)
  (ontable shot471)
  (dispenses dispenser419 ingredient379)
  (dispenses dispenser193 ingredient128)
  (dispenses dispenser324 ingredient295)
  (dispenses dispenser116 ingredient309)
  (clean shaker279)
  (clean shot38)
  (clean shot249)
  (clean shot471)
  (empty shaker279)
  (empty shot38)
  (empty shot249)
  (empty shot471)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker279 l0)
  (shaker-level shaker279 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail56 ingredient128)
  (cocktail-part2 cocktail56 ingredient379)
)
 (:goal
  (and
      (contains shot38 cocktail56)
      (contains shot249 ingredient379)
)))

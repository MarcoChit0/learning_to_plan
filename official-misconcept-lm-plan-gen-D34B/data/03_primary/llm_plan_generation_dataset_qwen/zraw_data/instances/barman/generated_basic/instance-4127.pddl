(define (problem prob)
 (:domain barman)
 (:objects 
      shaker297 - shaker
      left right - hand
      shot157 shot308 - shot
      ingredient478 ingredient417 ingredient383 - ingredient
      cocktail41 - cocktail
      dispenser210 dispenser193 dispenser24 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker297)
  (ontable shot157)
  (ontable shot308)
  (dispenses dispenser210 ingredient478)
  (dispenses dispenser193 ingredient417)
  (dispenses dispenser24 ingredient383)
  (clean shaker297)
  (clean shot157)
  (clean shot308)
  (empty shaker297)
  (empty shot157)
  (empty shot308)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker297 l0)
  (shaker-level shaker297 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail41 ingredient478)
  (cocktail-part2 cocktail41 ingredient417)
)
 (:goal
  (and
      (contains shot157 cocktail41)
)))

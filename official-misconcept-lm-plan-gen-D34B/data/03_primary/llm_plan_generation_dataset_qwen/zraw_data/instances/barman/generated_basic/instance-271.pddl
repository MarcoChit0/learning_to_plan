(define (problem prob)
 (:domain barman)
 (:objects 
      shaker64 - shaker
      left right - hand
      shot168 shot178 - shot
      ingredient35 ingredient128 ingredient259 - ingredient
      cocktail1 - cocktail
      dispenser32 dispenser327 dispenser86 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker64)
  (ontable shot168)
  (ontable shot178)
  (dispenses dispenser32 ingredient35)
  (dispenses dispenser327 ingredient128)
  (dispenses dispenser86 ingredient259)
  (clean shaker64)
  (clean shot168)
  (clean shot178)
  (empty shaker64)
  (empty shot168)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker64 l0)
  (shaker-level shaker64 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient35)
  (cocktail-part2 cocktail1 ingredient259)
)
 (:goal
  (and
      (contains shot168 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker381 - shaker
      left right - hand
      shot130 shot339 shot160 - shot
      ingredient321 ingredient249 ingredient199 ingredient90 - ingredient
      cocktail0 - cocktail
      dispenser161 dispenser50 dispenser79 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker381)
  (ontable shot130)
  (ontable shot339)
  (ontable shot160)
  (dispenses dispenser161 ingredient321)
  (dispenses dispenser50 ingredient249)
  (dispenses dispenser79 ingredient199)
  (dispenses dispenser92 ingredient90)
  (clean shaker381)
  (clean shot130)
  (clean shot339)
  (clean shot160)
  (empty shaker381)
  (empty shot130)
  (empty shot339)
  (empty shot160)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker381 l0)
  (shaker-level shaker381 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail0 ingredient199)
  (cocktail-part2 cocktail0 ingredient249)
)
 (:goal
  (and
      (contains shot130 cocktail0)
      (contains shot339 cocktail0)
)))

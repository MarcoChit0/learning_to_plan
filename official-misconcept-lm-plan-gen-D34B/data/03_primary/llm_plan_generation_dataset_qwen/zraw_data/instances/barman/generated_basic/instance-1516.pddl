(define (problem prob)
 (:domain barman)
 (:objects 
      shaker31 - shaker
      left right - hand
      shot39 shot369 shot382 shot443 - shot
      ingredient92 ingredient71 ingredient192 - ingredient
      cocktail284 - cocktail
      dispenser51 dispenser463 dispenser204 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker31)
  (ontable shot39)
  (ontable shot369)
  (ontable shot382)
  (ontable shot443)
  (dispenses dispenser51 ingredient92)
  (dispenses dispenser463 ingredient71)
  (dispenses dispenser204 ingredient192)
  (clean shaker31)
  (clean shot39)
  (clean shot369)
  (clean shot382)
  (clean shot443)
  (empty shaker31)
  (empty shot39)
  (empty shot369)
  (empty shot382)
  (empty shot443)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker31 l0)
  (shaker-level shaker31 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail284 ingredient71)
  (cocktail-part2 cocktail284 ingredient192)
)
 (:goal
  (and
      (contains shot39 cocktail284)
      (contains shot369 cocktail284)
      (contains shot382 ingredient92)
)))

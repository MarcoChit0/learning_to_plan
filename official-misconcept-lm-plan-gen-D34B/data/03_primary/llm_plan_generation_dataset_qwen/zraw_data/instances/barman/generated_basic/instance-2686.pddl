(define (problem prob)
 (:domain barman)
 (:objects 
      shaker167 - shaker
      left right - hand
      shot464 shot334 - shot
      ingredient131 ingredient412 ingredient332 - ingredient
      cocktail94 - cocktail
      dispenser433 dispenser26 dispenser490 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker167)
  (ontable shot464)
  (ontable shot334)
  (dispenses dispenser433 ingredient131)
  (dispenses dispenser26 ingredient412)
  (dispenses dispenser490 ingredient332)
  (clean shaker167)
  (clean shot464)
  (clean shot334)
  (empty shaker167)
  (empty shot464)
  (empty shot334)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker167 l0)
  (shaker-level shaker167 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail94 ingredient131)
  (cocktail-part2 cocktail94 ingredient332)
)
 (:goal
  (and
      (contains shot464 cocktail94)
)))

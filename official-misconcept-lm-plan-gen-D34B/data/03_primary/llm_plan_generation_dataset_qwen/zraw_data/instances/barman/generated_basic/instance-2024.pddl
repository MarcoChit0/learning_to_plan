(define (problem prob)
 (:domain barman)
 (:objects 
      shaker312 - shaker
      left right - hand
      shot189 shot321 - shot
      ingredient232 ingredient36 - ingredient
      cocktail268 - cocktail
      dispenser301 dispenser364 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker312)
  (ontable shot189)
  (ontable shot321)
  (dispenses dispenser301 ingredient232)
  (dispenses dispenser364 ingredient36)
  (clean shaker312)
  (clean shot189)
  (clean shot321)
  (empty shaker312)
  (empty shot189)
  (empty shot321)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker312 l0)
  (shaker-level shaker312 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient36)
  (cocktail-part2 cocktail268 ingredient232)
)
 (:goal
  (and
      (contains shot189 cocktail268)
)))

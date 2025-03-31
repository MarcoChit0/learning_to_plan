(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot397 shot348 shot345 - shot
      ingredient197 ingredient94 ingredient310 ingredient103 - ingredient
      cocktail298 - cocktail
      dispenser121 dispenser252 dispenser235 dispenser274 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot397)
  (ontable shot348)
  (ontable shot345)
  (dispenses dispenser121 ingredient197)
  (dispenses dispenser252 ingredient94)
  (dispenses dispenser235 ingredient310)
  (dispenses dispenser274 ingredient103)
  (clean shaker234)
  (clean shot397)
  (clean shot348)
  (clean shot345)
  (empty shaker234)
  (empty shot397)
  (empty shot348)
  (empty shot345)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail298 ingredient197)
  (cocktail-part2 cocktail298 ingredient103)
)
 (:goal
  (and
      (contains shot397 cocktail298)
      (contains shot348 ingredient94)
)))

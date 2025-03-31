(define (problem prob)
 (:domain barman)
 (:objects 
      shaker14 - shaker
      left right - hand
      shot214 shot266 - shot
      ingredient119 ingredient273 ingredient223 ingredient351 - ingredient
      cocktail357 - cocktail
      dispenser460 dispenser387 dispenser171 dispenser368 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker14)
  (ontable shot214)
  (ontable shot266)
  (dispenses dispenser460 ingredient119)
  (dispenses dispenser387 ingredient273)
  (dispenses dispenser171 ingredient223)
  (dispenses dispenser368 ingredient351)
  (clean shaker14)
  (clean shot214)
  (clean shot266)
  (empty shaker14)
  (empty shot214)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker14 l0)
  (shaker-level shaker14 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail357 ingredient119)
  (cocktail-part2 cocktail357 ingredient223)
)
 (:goal
  (and
      (contains shot214 cocktail357)
)))

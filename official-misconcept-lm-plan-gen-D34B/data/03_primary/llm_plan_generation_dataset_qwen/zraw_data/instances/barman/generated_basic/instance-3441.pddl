(define (problem prob)
 (:domain barman)
 (:objects 
      shaker388 - shaker
      left right - hand
      shot339 shot294 - shot
      ingredient206 ingredient429 ingredient440 ingredient201 - ingredient
      cocktail330 - cocktail
      dispenser339 dispenser431 dispenser277 dispenser427 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker388)
  (ontable shot339)
  (ontable shot294)
  (dispenses dispenser339 ingredient206)
  (dispenses dispenser431 ingredient429)
  (dispenses dispenser277 ingredient440)
  (dispenses dispenser427 ingredient201)
  (clean shaker388)
  (clean shot339)
  (clean shot294)
  (empty shaker388)
  (empty shot339)
  (empty shot294)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker388 l0)
  (shaker-level shaker388 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient440)
  (cocktail-part2 cocktail330 ingredient206)
)
 (:goal
  (and
      (contains shot339 cocktail330)
)))

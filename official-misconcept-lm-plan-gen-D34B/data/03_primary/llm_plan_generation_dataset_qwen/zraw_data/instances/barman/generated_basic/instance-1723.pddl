(define (problem prob)
 (:domain barman)
 (:objects 
      shaker68 - shaker
      left right - hand
      shot310 shot108 - shot
      ingredient55 ingredient186 ingredient304 ingredient206 - ingredient
      cocktail326 - cocktail
      dispenser45 dispenser356 dispenser386 dispenser297 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker68)
  (ontable shot310)
  (ontable shot108)
  (dispenses dispenser45 ingredient55)
  (dispenses dispenser356 ingredient186)
  (dispenses dispenser386 ingredient304)
  (dispenses dispenser297 ingredient206)
  (clean shaker68)
  (clean shot310)
  (clean shot108)
  (empty shaker68)
  (empty shot310)
  (empty shot108)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker68 l0)
  (shaker-level shaker68 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail326 ingredient55)
  (cocktail-part2 cocktail326 ingredient304)
)
 (:goal
  (and
      (contains shot310 cocktail326)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker292 - shaker
      left right - hand
      shot220 shot129 shot263 - shot
      ingredient220 ingredient206 - ingredient
      cocktail38 - cocktail
      dispenser119 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker292)
  (ontable shot220)
  (ontable shot129)
  (ontable shot263)
  (dispenses dispenser119 ingredient220)
  (dispenses dispenser306 ingredient206)
  (clean shaker292)
  (clean shot220)
  (clean shot129)
  (clean shot263)
  (empty shaker292)
  (empty shot220)
  (empty shot129)
  (empty shot263)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker292 l0)
  (shaker-level shaker292 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail38 ingredient220)
  (cocktail-part2 cocktail38 ingredient206)
)
 (:goal
  (and
      (contains shot220 cocktail38)
      (contains shot129 ingredient206)
)))

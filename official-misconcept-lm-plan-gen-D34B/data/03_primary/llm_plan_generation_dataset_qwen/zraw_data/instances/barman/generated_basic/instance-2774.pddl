(define (problem prob)
 (:domain barman)
 (:objects 
      shaker210 - shaker
      left right - hand
      shot450 shot428 - shot
      ingredient493 ingredient206 ingredient65 - ingredient
      cocktail171 - cocktail
      dispenser429 dispenser455 dispenser450 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker210)
  (ontable shot450)
  (ontable shot428)
  (dispenses dispenser429 ingredient493)
  (dispenses dispenser455 ingredient206)
  (dispenses dispenser450 ingredient65)
  (clean shaker210)
  (clean shot450)
  (clean shot428)
  (empty shaker210)
  (empty shot450)
  (empty shot428)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker210 l0)
  (shaker-level shaker210 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail171 ingredient493)
  (cocktail-part2 cocktail171 ingredient206)
)
 (:goal
  (and
      (contains shot450 cocktail171)
)))

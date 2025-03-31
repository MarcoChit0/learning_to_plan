(define (problem prob)
 (:domain barman)
 (:objects 
      shaker424 - shaker
      left right - hand
      shot307 shot255 - shot
      ingredient247 ingredient20 ingredient250 - ingredient
      cocktail1 - cocktail
      dispenser457 dispenser253 dispenser279 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker424)
  (ontable shot307)
  (ontable shot255)
  (dispenses dispenser457 ingredient247)
  (dispenses dispenser253 ingredient20)
  (dispenses dispenser279 ingredient250)
  (clean shaker424)
  (clean shot307)
  (clean shot255)
  (empty shaker424)
  (empty shot307)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker424 l0)
  (shaker-level shaker424 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient250)
  (cocktail-part2 cocktail1 ingredient20)
)
 (:goal
  (and
      (contains shot307 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot58 shot265 - shot
      ingredient454 ingredient206 ingredient215 - ingredient
      cocktail1 - cocktail
      dispenser354 dispenser80 dispenser18 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot58)
  (ontable shot265)
  (dispenses dispenser354 ingredient454)
  (dispenses dispenser80 ingredient206)
  (dispenses dispenser18 ingredient215)
  (clean shaker416)
  (clean shot58)
  (clean shot265)
  (empty shaker416)
  (empty shot58)
  (empty shot265)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient206)
  (cocktail-part2 cocktail1 ingredient454)
)
 (:goal
  (and
      (contains shot58 cocktail1)
)))

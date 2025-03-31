(define (problem prob)
 (:domain barman)
 (:objects 
      shaker90 - shaker
      left right - hand
      shot266 - shot
      ingredient215 ingredient29 - ingredient
      cocktail1 - cocktail
      dispenser154 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker90)
  (ontable shot266)
  (dispenses dispenser154 ingredient215)
  (dispenses dispenser206 ingredient29)
  (clean shaker90)
  (clean shot266)
  (empty shaker90)
  (empty shot266)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker90 l0)
  (shaker-level shaker90 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient29)
  (cocktail-part2 cocktail1 ingredient215)
)
 (:goal
  (and
      (contains shot266 cocktail1)
)))

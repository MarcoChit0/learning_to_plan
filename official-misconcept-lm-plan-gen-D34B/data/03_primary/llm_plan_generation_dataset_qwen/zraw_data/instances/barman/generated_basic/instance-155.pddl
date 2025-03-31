(define (problem prob)
 (:domain barman)
 (:objects 
      shaker212 - shaker
      left right - hand
      shot131 shot339 - shot
      ingredient359 ingredient128 ingredient80 - ingredient
      cocktail1 - cocktail
      dispenser406 dispenser308 dispenser231 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker212)
  (ontable shot131)
  (ontable shot339)
  (dispenses dispenser406 ingredient359)
  (dispenses dispenser308 ingredient128)
  (dispenses dispenser231 ingredient80)
  (clean shaker212)
  (clean shot131)
  (clean shot339)
  (empty shaker212)
  (empty shot131)
  (empty shot339)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker212 l0)
  (shaker-level shaker212 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient80)
  (cocktail-part2 cocktail1 ingredient359)
)
 (:goal
  (and
      (contains shot131 cocktail1)
)))

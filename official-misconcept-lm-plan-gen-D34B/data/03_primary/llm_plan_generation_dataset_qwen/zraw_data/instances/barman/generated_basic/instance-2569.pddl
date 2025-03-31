(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot325 shot211 shot433 - shot
      ingredient6 ingredient135 - ingredient
      cocktail469 - cocktail
      dispenser219 dispenser177 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot325)
  (ontable shot211)
  (ontable shot433)
  (dispenses dispenser219 ingredient6)
  (dispenses dispenser177 ingredient135)
  (clean shaker454)
  (clean shot325)
  (clean shot211)
  (clean shot433)
  (empty shaker454)
  (empty shot325)
  (empty shot211)
  (empty shot433)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail469 ingredient6)
  (cocktail-part2 cocktail469 ingredient135)
)
 (:goal
  (and
      (contains shot325 cocktail469)
      (contains shot211 ingredient6)
)))

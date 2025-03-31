(define (problem prob)
 (:domain barman)
 (:objects 
      shaker197 - shaker
      left right - hand
      shot487 shot415 - shot
      ingredient303 ingredient224 ingredient228 ingredient123 - ingredient
      cocktail1 - cocktail
      dispenser258 dispenser217 dispenser386 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker197)
  (ontable shot487)
  (ontable shot415)
  (dispenses dispenser258 ingredient303)
  (dispenses dispenser217 ingredient224)
  (dispenses dispenser386 ingredient228)
  (dispenses dispenser361 ingredient123)
  (clean shaker197)
  (clean shot487)
  (clean shot415)
  (empty shaker197)
  (empty shot487)
  (empty shot415)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker197 l0)
  (shaker-level shaker197 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient303)
  (cocktail-part2 cocktail1 ingredient224)
)
 (:goal
  (and
      (contains shot487 cocktail1)
)))

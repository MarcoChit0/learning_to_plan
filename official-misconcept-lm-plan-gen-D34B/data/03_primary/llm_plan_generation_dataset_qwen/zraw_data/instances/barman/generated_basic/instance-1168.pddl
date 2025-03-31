(define (problem prob)
 (:domain barman)
 (:objects 
      shaker382 - shaker
      left right - hand
      shot86 shot186 - shot
      ingredient212 ingredient99 ingredient54 - ingredient
      cocktail1 - cocktail
      dispenser279 dispenser53 dispenser361 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker382)
  (ontable shot86)
  (ontable shot186)
  (dispenses dispenser279 ingredient212)
  (dispenses dispenser53 ingredient99)
  (dispenses dispenser361 ingredient54)
  (clean shaker382)
  (clean shot86)
  (clean shot186)
  (empty shaker382)
  (empty shot86)
  (empty shot186)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker382 l0)
  (shaker-level shaker382 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient54)
  (cocktail-part2 cocktail1 ingredient99)
)
 (:goal
  (and
      (contains shot86 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot121 - shot
      ingredient465 ingredient99 ingredient331 - ingredient
      cocktail392 - cocktail
      dispenser287 dispenser177 dispenser52 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot121)
  (dispenses dispenser287 ingredient465)
  (dispenses dispenser177 ingredient99)
  (dispenses dispenser52 ingredient331)
  (clean shaker454)
  (clean shot121)
  (empty shaker454)
  (empty shot121)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail392 ingredient99)
  (cocktail-part2 cocktail392 ingredient465)
)
 (:goal
  (and
      (contains shot121 cocktail392)
)))

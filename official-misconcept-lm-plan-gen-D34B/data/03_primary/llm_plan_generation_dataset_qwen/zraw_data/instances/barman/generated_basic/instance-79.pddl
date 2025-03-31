(define (problem prob)
 (:domain barman)
 (:objects 
      shaker51 - shaker
      left right - hand
      shot31 shot454 - shot
      ingredient96 ingredient407 ingredient31 - ingredient
      cocktail1 - cocktail
      dispenser302 dispenser409 dispenser74 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker51)
  (ontable shot31)
  (ontable shot454)
  (dispenses dispenser302 ingredient96)
  (dispenses dispenser409 ingredient407)
  (dispenses dispenser74 ingredient31)
  (clean shaker51)
  (clean shot31)
  (clean shot454)
  (empty shaker51)
  (empty shot31)
  (empty shot454)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker51 l0)
  (shaker-level shaker51 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient31)
  (cocktail-part2 cocktail1 ingredient407)
)
 (:goal
  (and
      (contains shot31 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot11 - shot
      ingredient258 ingredient481 - ingredient
      cocktail88 - cocktail
      dispenser271 dispenser194 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot11)
  (dispenses dispenser271 ingredient258)
  (dispenses dispenser194 ingredient481)
  (clean shaker454)
  (clean shot11)
  (empty shaker454)
  (empty shot11)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient481)
  (cocktail-part2 cocktail88 ingredient258)
)
 (:goal
  (and
      (contains shot11 cocktail88)
)))

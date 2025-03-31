(define (problem prob)
 (:domain barman)
 (:objects 
      shaker454 - shaker
      left right - hand
      shot420 - shot
      ingredient437 ingredient56 - ingredient
      cocktail95 - cocktail
      dispenser175 dispenser131 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker454)
  (ontable shot420)
  (dispenses dispenser175 ingredient437)
  (dispenses dispenser131 ingredient56)
  (clean shaker454)
  (clean shot420)
  (empty shaker454)
  (empty shot420)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker454 l0)
  (shaker-level shaker454 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail95 ingredient56)
  (cocktail-part2 cocktail95 ingredient437)
)
 (:goal
  (and
      (contains shot420 cocktail95)
)))

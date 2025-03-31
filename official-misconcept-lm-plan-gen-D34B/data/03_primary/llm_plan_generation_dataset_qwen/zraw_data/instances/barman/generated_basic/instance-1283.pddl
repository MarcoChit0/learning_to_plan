(define (problem prob)
 (:domain barman)
 (:objects 
      shaker313 - shaker
      left right - hand
      shot33 - shot
      ingredient212 ingredient493 ingredient454 ingredient425 - ingredient
      cocktail315 - cocktail
      dispenser426 dispenser175 dispenser250 dispenser328 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker313)
  (ontable shot33)
  (dispenses dispenser426 ingredient212)
  (dispenses dispenser175 ingredient493)
  (dispenses dispenser250 ingredient454)
  (dispenses dispenser328 ingredient425)
  (clean shaker313)
  (clean shot33)
  (empty shaker313)
  (empty shot33)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker313 l0)
  (shaker-level shaker313 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail315 ingredient454)
  (cocktail-part2 cocktail315 ingredient212)
)
 (:goal
  (and
      (contains shot33 cocktail315)
)))

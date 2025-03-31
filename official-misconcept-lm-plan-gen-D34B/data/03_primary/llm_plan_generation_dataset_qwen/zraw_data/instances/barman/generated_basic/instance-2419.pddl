(define (problem prob)
 (:domain barman)
 (:objects 
      shaker447 - shaker
      left right - hand
      shot63 - shot
      ingredient494 ingredient22 - ingredient
      cocktail17 - cocktail
      dispenser457 dispenser46 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker447)
  (ontable shot63)
  (dispenses dispenser457 ingredient494)
  (dispenses dispenser46 ingredient22)
  (clean shaker447)
  (clean shot63)
  (empty shaker447)
  (empty shot63)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker447 l0)
  (shaker-level shaker447 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail17 ingredient494)
  (cocktail-part2 cocktail17 ingredient22)
)
 (:goal
  (and
      (contains shot63 cocktail17)
)))

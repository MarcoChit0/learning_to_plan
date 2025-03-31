(define (problem prob)
 (:domain barman)
 (:objects 
      shaker39 - shaker
      left right - hand
      shot337 - shot
      ingredient57 ingredient79 - ingredient
      cocktail70 - cocktail
      dispenser67 dispenser315 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker39)
  (ontable shot337)
  (dispenses dispenser67 ingredient57)
  (dispenses dispenser315 ingredient79)
  (clean shaker39)
  (clean shot337)
  (empty shaker39)
  (empty shot337)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker39 l0)
  (shaker-level shaker39 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail70 ingredient79)
  (cocktail-part2 cocktail70 ingredient57)
)
 (:goal
  (and
      (contains shot337 cocktail70)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker115 - shaker
      left right - hand
      shot110 - shot
      ingredient67 ingredient212 ingredient233 ingredient309 - ingredient
      cocktail335 - cocktail
      dispenser53 dispenser184 dispenser91 dispenser392 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker115)
  (ontable shot110)
  (dispenses dispenser53 ingredient67)
  (dispenses dispenser184 ingredient212)
  (dispenses dispenser91 ingredient233)
  (dispenses dispenser392 ingredient309)
  (clean shaker115)
  (clean shot110)
  (empty shaker115)
  (empty shot110)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker115 l0)
  (shaker-level shaker115 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail335 ingredient212)
  (cocktail-part2 cocktail335 ingredient233)
)
 (:goal
  (and
      (contains shot110 cocktail335)
)))

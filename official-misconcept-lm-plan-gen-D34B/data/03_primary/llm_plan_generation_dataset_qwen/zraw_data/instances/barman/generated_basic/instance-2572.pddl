(define (problem prob)
 (:domain barman)
 (:objects 
      shaker241 - shaker
      left right - hand
      shot178 - shot
      ingredient490 ingredient55 - ingredient
      cocktail31 - cocktail
      dispenser149 dispenser305 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker241)
  (ontable shot178)
  (dispenses dispenser149 ingredient490)
  (dispenses dispenser305 ingredient55)
  (clean shaker241)
  (clean shot178)
  (empty shaker241)
  (empty shot178)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker241 l0)
  (shaker-level shaker241 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail31 ingredient490)
  (cocktail-part2 cocktail31 ingredient55)
)
 (:goal
  (and
      (contains shot178 cocktail31)
)))

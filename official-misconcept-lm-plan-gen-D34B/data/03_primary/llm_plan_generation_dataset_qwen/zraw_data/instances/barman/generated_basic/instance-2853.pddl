(define (problem prob)
 (:domain barman)
 (:objects 
      shaker117 - shaker
      left right - hand
      shot178 shot135 - shot
      ingredient170 ingredient393 ingredient55 ingredient304 - ingredient
      cocktail88 - cocktail
      dispenser457 dispenser54 dispenser40 dispenser170 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker117)
  (ontable shot178)
  (ontable shot135)
  (dispenses dispenser457 ingredient170)
  (dispenses dispenser54 ingredient393)
  (dispenses dispenser40 ingredient55)
  (dispenses dispenser170 ingredient304)
  (clean shaker117)
  (clean shot178)
  (clean shot135)
  (empty shaker117)
  (empty shot178)
  (empty shot135)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker117 l0)
  (shaker-level shaker117 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail88 ingredient55)
  (cocktail-part2 cocktail88 ingredient393)
)
 (:goal
  (and
      (contains shot178 cocktail88)
)))

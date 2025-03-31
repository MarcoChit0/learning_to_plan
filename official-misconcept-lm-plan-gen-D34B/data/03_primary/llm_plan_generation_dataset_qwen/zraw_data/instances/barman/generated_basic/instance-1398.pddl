(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot420 - shot
      ingredient408 ingredient26 ingredient166 ingredient241 - ingredient
      cocktail365 - cocktail
      dispenser306 dispenser164 dispenser349 dispenser114 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot420)
  (dispenses dispenser306 ingredient408)
  (dispenses dispenser164 ingredient26)
  (dispenses dispenser349 ingredient166)
  (dispenses dispenser114 ingredient241)
  (clean shaker296)
  (clean shot420)
  (empty shaker296)
  (empty shot420)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail365 ingredient166)
  (cocktail-part2 cocktail365 ingredient26)
)
 (:goal
  (and
      (contains shot420 cocktail365)
)))

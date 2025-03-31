(define (problem prob)
 (:domain barman)
 (:objects 
      shaker247 - shaker
      left right - hand
      shot50 - shot
      ingredient9 ingredient364 ingredient298 - ingredient
      cocktail375 - cocktail
      dispenser91 dispenser127 dispenser320 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker247)
  (ontable shot50)
  (dispenses dispenser91 ingredient9)
  (dispenses dispenser127 ingredient364)
  (dispenses dispenser320 ingredient298)
  (clean shaker247)
  (clean shot50)
  (empty shaker247)
  (empty shot50)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker247 l0)
  (shaker-level shaker247 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail375 ingredient298)
  (cocktail-part2 cocktail375 ingredient364)
)
 (:goal
  (and
      (contains shot50 cocktail375)
)))

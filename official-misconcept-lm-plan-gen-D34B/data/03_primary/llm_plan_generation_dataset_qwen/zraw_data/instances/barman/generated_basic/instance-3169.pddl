(define (problem prob)
 (:domain barman)
 (:objects 
      shaker88 - shaker
      left right - hand
      shot196 - shot
      ingredient76 ingredient158 ingredient89 - ingredient
      cocktail416 - cocktail
      dispenser433 dispenser343 dispenser257 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker88)
  (ontable shot196)
  (dispenses dispenser433 ingredient76)
  (dispenses dispenser343 ingredient158)
  (dispenses dispenser257 ingredient89)
  (clean shaker88)
  (clean shot196)
  (empty shaker88)
  (empty shot196)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker88 l0)
  (shaker-level shaker88 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail416 ingredient158)
  (cocktail-part2 cocktail416 ingredient89)
)
 (:goal
  (and
      (contains shot196 cocktail416)
)))

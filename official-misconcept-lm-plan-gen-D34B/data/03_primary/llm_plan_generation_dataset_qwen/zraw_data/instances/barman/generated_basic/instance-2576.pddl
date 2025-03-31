(define (problem prob)
 (:domain barman)
 (:objects 
      shaker459 - shaker
      left right - hand
      shot79 - shot
      ingredient21 ingredient249 - ingredient
      cocktail159 - cocktail
      dispenser245 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker459)
  (ontable shot79)
  (dispenses dispenser245 ingredient21)
  (dispenses dispenser101 ingredient249)
  (clean shaker459)
  (clean shot79)
  (empty shaker459)
  (empty shot79)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker459 l0)
  (shaker-level shaker459 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail159 ingredient21)
  (cocktail-part2 cocktail159 ingredient249)
)
 (:goal
  (and
      (contains shot79 cocktail159)
)))

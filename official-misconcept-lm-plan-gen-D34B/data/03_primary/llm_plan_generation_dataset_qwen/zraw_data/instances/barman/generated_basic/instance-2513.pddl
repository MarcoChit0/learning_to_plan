(define (problem prob)
 (:domain barman)
 (:objects 
      shaker198 - shaker
      left right - hand
      shot60 - shot
      ingredient233 ingredient88 - ingredient
      cocktail181 - cocktail
      dispenser338 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker198)
  (ontable shot60)
  (dispenses dispenser338 ingredient233)
  (dispenses dispenser92 ingredient88)
  (clean shaker198)
  (clean shot60)
  (empty shaker198)
  (empty shot60)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker198 l0)
  (shaker-level shaker198 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail181 ingredient88)
  (cocktail-part2 cocktail181 ingredient233)
)
 (:goal
  (and
      (contains shot60 cocktail181)
)))

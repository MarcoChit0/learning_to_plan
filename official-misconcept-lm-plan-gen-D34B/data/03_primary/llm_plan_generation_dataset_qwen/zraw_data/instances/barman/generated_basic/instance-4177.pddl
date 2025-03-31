(define (problem prob)
 (:domain barman)
 (:objects 
      shaker303 - shaker
      left right - hand
      shot380 - shot
      ingredient15 ingredient248 ingredient488 ingredient362 - ingredient
      cocktail360 - cocktail
      dispenser184 dispenser203 dispenser56 dispenser426 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker303)
  (ontable shot380)
  (dispenses dispenser184 ingredient15)
  (dispenses dispenser203 ingredient248)
  (dispenses dispenser56 ingredient488)
  (dispenses dispenser426 ingredient362)
  (clean shaker303)
  (clean shot380)
  (empty shaker303)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker303 l0)
  (shaker-level shaker303 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail360 ingredient248)
  (cocktail-part2 cocktail360 ingredient362)
)
 (:goal
  (and
      (contains shot380 cocktail360)
)))

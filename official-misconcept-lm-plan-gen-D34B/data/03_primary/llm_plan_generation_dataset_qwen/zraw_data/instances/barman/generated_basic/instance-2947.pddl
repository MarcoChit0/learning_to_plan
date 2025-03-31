(define (problem prob)
 (:domain barman)
 (:objects 
      shaker58 - shaker
      left right - hand
      shot170 - shot
      ingredient222 ingredient78 ingredient65 ingredient410 - ingredient
      cocktail281 - cocktail
      dispenser317 dispenser359 dispenser137 dispenser379 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker58)
  (ontable shot170)
  (dispenses dispenser317 ingredient222)
  (dispenses dispenser359 ingredient78)
  (dispenses dispenser137 ingredient65)
  (dispenses dispenser379 ingredient410)
  (clean shaker58)
  (clean shot170)
  (empty shaker58)
  (empty shot170)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker58 l0)
  (shaker-level shaker58 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail281 ingredient78)
  (cocktail-part2 cocktail281 ingredient410)
)
 (:goal
  (and
      (contains shot170 cocktail281)
)))

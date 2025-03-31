(define (problem prob)
 (:domain barman)
 (:objects 
      shaker163 - shaker
      left right - hand
      shot373 shot42 shot431 - shot
      ingredient15 ingredient159 ingredient442 - ingredient
      cocktail499 - cocktail
      dispenser426 dispenser213 dispenser375 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker163)
  (ontable shot373)
  (ontable shot42)
  (ontable shot431)
  (dispenses dispenser426 ingredient15)
  (dispenses dispenser213 ingredient159)
  (dispenses dispenser375 ingredient442)
  (clean shaker163)
  (clean shot373)
  (clean shot42)
  (clean shot431)
  (empty shaker163)
  (empty shot373)
  (empty shot42)
  (empty shot431)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker163 l0)
  (shaker-level shaker163 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail499 ingredient15)
  (cocktail-part2 cocktail499 ingredient159)
)
 (:goal
  (and
      (contains shot373 cocktail499)
      (contains shot42 cocktail499)
)))

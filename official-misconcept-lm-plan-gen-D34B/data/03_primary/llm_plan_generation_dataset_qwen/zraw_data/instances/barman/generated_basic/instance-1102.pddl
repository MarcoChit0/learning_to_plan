(define (problem prob)
 (:domain barman)
 (:objects 
      shaker449 - shaker
      left right - hand
      shot292 - shot
      ingredient338 ingredient117 ingredient413 - ingredient
      cocktail1 - cocktail
      dispenser64 dispenser10 dispenser213 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker449)
  (ontable shot292)
  (dispenses dispenser64 ingredient338)
  (dispenses dispenser10 ingredient117)
  (dispenses dispenser213 ingredient413)
  (clean shaker449)
  (clean shot292)
  (empty shaker449)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker449 l0)
  (shaker-level shaker449 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient117)
  (cocktail-part2 cocktail1 ingredient338)
)
 (:goal
  (and
      (contains shot292 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot259 - shot
      ingredient36 ingredient359 ingredient338 - ingredient
      cocktail350 - cocktail
      dispenser319 dispenser74 dispenser168 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot259)
  (dispenses dispenser319 ingredient36)
  (dispenses dispenser74 ingredient359)
  (dispenses dispenser168 ingredient338)
  (clean shaker405)
  (clean shot259)
  (empty shaker405)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail350 ingredient36)
  (cocktail-part2 cocktail350 ingredient338)
)
 (:goal
  (and
      (contains shot259 cocktail350)
)))

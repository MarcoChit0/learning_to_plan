(define (problem prob)
 (:domain barman)
 (:objects 
      shaker42 - shaker
      left right - hand
      shot346 - shot
      ingredient144 ingredient233 ingredient152 ingredient429 - ingredient
      cocktail1 - cocktail
      dispenser381 dispenser235 dispenser346 dispenser275 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker42)
  (ontable shot346)
  (dispenses dispenser381 ingredient144)
  (dispenses dispenser235 ingredient233)
  (dispenses dispenser346 ingredient152)
  (dispenses dispenser275 ingredient429)
  (clean shaker42)
  (clean shot346)
  (empty shaker42)
  (empty shot346)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker42 l0)
  (shaker-level shaker42 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient152)
  (cocktail-part2 cocktail1 ingredient144)
)
 (:goal
  (and
      (contains shot346 cocktail1)
)))

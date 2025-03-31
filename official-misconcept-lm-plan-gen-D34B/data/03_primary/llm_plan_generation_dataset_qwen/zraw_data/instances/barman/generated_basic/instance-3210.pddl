(define (problem prob)
 (:domain barman)
 (:objects 
      shaker436 - shaker
      left right - hand
      shot19 - shot
      ingredient273 ingredient284 - ingredient
      cocktail40 - cocktail
      dispenser311 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker436)
  (ontable shot19)
  (dispenses dispenser311 ingredient273)
  (dispenses dispenser188 ingredient284)
  (clean shaker436)
  (clean shot19)
  (empty shaker436)
  (empty shot19)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker436 l0)
  (shaker-level shaker436 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail40 ingredient273)
  (cocktail-part2 cocktail40 ingredient284)
)
 (:goal
  (and
      (contains shot19 cocktail40)
)))

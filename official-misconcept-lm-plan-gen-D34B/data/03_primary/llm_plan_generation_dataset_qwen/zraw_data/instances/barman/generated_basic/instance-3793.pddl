(define (problem prob)
 (:domain barman)
 (:objects 
      shaker498 - shaker
      left right - hand
      shot126 - shot
      ingredient474 ingredient169 ingredient258 ingredient491 - ingredient
      cocktail331 - cocktail
      dispenser75 dispenser179 dispenser294 dispenser189 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker498)
  (ontable shot126)
  (dispenses dispenser75 ingredient474)
  (dispenses dispenser179 ingredient169)
  (dispenses dispenser294 ingredient258)
  (dispenses dispenser189 ingredient491)
  (clean shaker498)
  (clean shot126)
  (empty shaker498)
  (empty shot126)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker498 l0)
  (shaker-level shaker498 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail331 ingredient491)
  (cocktail-part2 cocktail331 ingredient474)
)
 (:goal
  (and
      (contains shot126 cocktail331)
)))

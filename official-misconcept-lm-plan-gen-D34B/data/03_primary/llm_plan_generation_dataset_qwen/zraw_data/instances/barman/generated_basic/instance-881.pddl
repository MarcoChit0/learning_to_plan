(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot43 - shot
      ingredient195 ingredient421 - ingredient
      cocktail1 - cocktail
      dispenser77 dispenser312 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot43)
  (dispenses dispenser77 ingredient195)
  (dispenses dispenser312 ingredient421)
  (clean shaker474)
  (clean shot43)
  (empty shaker474)
  (empty shot43)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient195)
  (cocktail-part2 cocktail1 ingredient421)
)
 (:goal
  (and
      (contains shot43 cocktail1)
)))

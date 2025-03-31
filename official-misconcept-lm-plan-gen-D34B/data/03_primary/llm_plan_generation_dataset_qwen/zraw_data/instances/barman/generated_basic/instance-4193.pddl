(define (problem prob)
 (:domain barman)
 (:objects 
      shaker474 - shaker
      left right - hand
      shot165 - shot
      ingredient443 ingredient310 ingredient394 ingredient326 - ingredient
      cocktail475 - cocktail
      dispenser220 dispenser396 dispenser373 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker474)
  (ontable shot165)
  (dispenses dispenser220 ingredient443)
  (dispenses dispenser396 ingredient310)
  (dispenses dispenser373 ingredient394)
  (dispenses dispenser437 ingredient326)
  (clean shaker474)
  (clean shot165)
  (empty shaker474)
  (empty shot165)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker474 l0)
  (shaker-level shaker474 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail475 ingredient394)
  (cocktail-part2 cocktail475 ingredient310)
)
 (:goal
  (and
      (contains shot165 cocktail475)
)))

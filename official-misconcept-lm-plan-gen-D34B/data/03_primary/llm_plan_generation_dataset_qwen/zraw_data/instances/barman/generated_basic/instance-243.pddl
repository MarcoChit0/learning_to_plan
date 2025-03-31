(define (problem prob)
 (:domain barman)
 (:objects 
      shaker457 - shaker
      left right - hand
      shot220 - shot
      ingredient52 ingredient365 - ingredient
      cocktail1 - cocktail
      dispenser428 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker457)
  (ontable shot220)
  (dispenses dispenser428 ingredient52)
  (dispenses dispenser128 ingredient365)
  (clean shaker457)
  (clean shot220)
  (empty shaker457)
  (empty shot220)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker457 l0)
  (shaker-level shaker457 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient52)
  (cocktail-part2 cocktail1 ingredient365)
)
 (:goal
  (and
      (contains shot220 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker273 - shaker
      left right - hand
      shot203 - shot
      ingredient451 ingredient200 ingredient133 - ingredient
      cocktail106 - cocktail
      dispenser270 dispenser110 dispenser444 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker273)
  (ontable shot203)
  (dispenses dispenser270 ingredient451)
  (dispenses dispenser110 ingredient200)
  (dispenses dispenser444 ingredient133)
  (clean shaker273)
  (clean shot203)
  (empty shaker273)
  (empty shot203)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker273 l0)
  (shaker-level shaker273 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail106 ingredient200)
  (cocktail-part2 cocktail106 ingredient133)
)
 (:goal
  (and
      (contains shot203 cocktail106)
)))

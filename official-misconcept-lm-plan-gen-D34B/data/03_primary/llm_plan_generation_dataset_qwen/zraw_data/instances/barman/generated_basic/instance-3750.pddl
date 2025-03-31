(define (problem prob)
 (:domain barman)
 (:objects 
      shaker372 - shaker
      left right - hand
      shot194 shot391 shot274 - shot
      ingredient378 ingredient366 ingredient483 - ingredient
      cocktail261 - cocktail
      dispenser137 dispenser473 dispenser324 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker372)
  (ontable shot194)
  (ontable shot391)
  (ontable shot274)
  (dispenses dispenser137 ingredient378)
  (dispenses dispenser473 ingredient366)
  (dispenses dispenser324 ingredient483)
  (clean shaker372)
  (clean shot194)
  (clean shot391)
  (clean shot274)
  (empty shaker372)
  (empty shot194)
  (empty shot391)
  (empty shot274)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker372 l0)
  (shaker-level shaker372 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail261 ingredient378)
  (cocktail-part2 cocktail261 ingredient366)
)
 (:goal
  (and
      (contains shot194 cocktail261)
      (contains shot391 ingredient366)
)))

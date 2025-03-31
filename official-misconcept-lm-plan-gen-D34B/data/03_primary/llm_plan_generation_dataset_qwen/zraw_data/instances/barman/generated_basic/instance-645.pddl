(define (problem prob)
 (:domain barman)
 (:objects 
      shaker237 - shaker
      left right - hand
      shot41 shot380 - shot
      ingredient454 ingredient428 ingredient397 - ingredient
      cocktail1 - cocktail
      dispenser486 dispenser142 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker237)
  (ontable shot41)
  (ontable shot380)
  (dispenses dispenser486 ingredient454)
  (dispenses dispenser142 ingredient428)
  (dispenses dispenser73 ingredient397)
  (clean shaker237)
  (clean shot41)
  (clean shot380)
  (empty shaker237)
  (empty shot41)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker237 l0)
  (shaker-level shaker237 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient397)
  (cocktail-part2 cocktail1 ingredient428)
)
 (:goal
  (and
      (contains shot41 cocktail1)
)))

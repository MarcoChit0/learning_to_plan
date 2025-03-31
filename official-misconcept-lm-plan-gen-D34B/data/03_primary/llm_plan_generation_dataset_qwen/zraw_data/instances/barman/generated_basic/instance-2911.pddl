(define (problem prob)
 (:domain barman)
 (:objects 
      shaker272 - shaker
      left right - hand
      shot380 - shot
      ingredient458 ingredient397 - ingredient
      cocktail390 - cocktail
      dispenser49 dispenser256 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker272)
  (ontable shot380)
  (dispenses dispenser49 ingredient458)
  (dispenses dispenser256 ingredient397)
  (clean shaker272)
  (clean shot380)
  (empty shaker272)
  (empty shot380)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker272 l0)
  (shaker-level shaker272 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail390 ingredient458)
  (cocktail-part2 cocktail390 ingredient397)
)
 (:goal
  (and
      (contains shot380 cocktail390)
)))

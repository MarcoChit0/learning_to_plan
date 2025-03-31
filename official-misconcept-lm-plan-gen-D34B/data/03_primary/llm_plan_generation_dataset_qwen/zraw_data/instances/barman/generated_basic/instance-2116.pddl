(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot184 shot37 shot343 - shot
      ingredient60 ingredient226 - ingredient
      cocktail409 - cocktail
      dispenser280 dispenser92 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot184)
  (ontable shot37)
  (ontable shot343)
  (dispenses dispenser280 ingredient60)
  (dispenses dispenser92 ingredient226)
  (clean shaker336)
  (clean shot184)
  (clean shot37)
  (clean shot343)
  (empty shaker336)
  (empty shot184)
  (empty shot37)
  (empty shot343)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail409 ingredient60)
  (cocktail-part2 cocktail409 ingredient226)
)
 (:goal
  (and
      (contains shot184 cocktail409)
      (contains shot37 cocktail409)
)))

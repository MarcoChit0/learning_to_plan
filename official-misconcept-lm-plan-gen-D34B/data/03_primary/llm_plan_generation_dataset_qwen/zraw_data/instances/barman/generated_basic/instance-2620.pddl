(define (problem prob)
 (:domain barman)
 (:objects 
      shaker472 - shaker
      left right - hand
      shot141 shot57 - shot
      ingredient40 ingredient92 ingredient215 - ingredient
      cocktail455 - cocktail
      dispenser338 dispenser251 dispenser220 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker472)
  (ontable shot141)
  (ontable shot57)
  (dispenses dispenser338 ingredient40)
  (dispenses dispenser251 ingredient92)
  (dispenses dispenser220 ingredient215)
  (clean shaker472)
  (clean shot141)
  (clean shot57)
  (empty shaker472)
  (empty shot141)
  (empty shot57)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker472 l0)
  (shaker-level shaker472 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail455 ingredient92)
  (cocktail-part2 cocktail455 ingredient40)
)
 (:goal
  (and
      (contains shot141 cocktail455)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker185 - shaker
      left right - hand
      shot212 - shot
      ingredient478 ingredient101 ingredient296 ingredient7 - ingredient
      cocktail78 - cocktail
      dispenser41 dispenser264 dispenser103 dispenser206 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker185)
  (ontable shot212)
  (dispenses dispenser41 ingredient478)
  (dispenses dispenser264 ingredient101)
  (dispenses dispenser103 ingredient296)
  (dispenses dispenser206 ingredient7)
  (clean shaker185)
  (clean shot212)
  (empty shaker185)
  (empty shot212)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker185 l0)
  (shaker-level shaker185 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail78 ingredient296)
  (cocktail-part2 cocktail78 ingredient478)
)
 (:goal
  (and
      (contains shot212 cocktail78)
)))

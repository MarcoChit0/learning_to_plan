(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot343 shot229 - shot
      ingredient478 ingredient152 - ingredient
      cocktail1 - cocktail
      dispenser357 dispenser287 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot343)
  (ontable shot229)
  (dispenses dispenser357 ingredient478)
  (dispenses dispenser287 ingredient152)
  (clean shaker4)
  (clean shot343)
  (clean shot229)
  (empty shaker4)
  (empty shot343)
  (empty shot229)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient152)
  (cocktail-part2 cocktail1 ingredient478)
)
 (:goal
  (and
      (contains shot343 cocktail1)
)))

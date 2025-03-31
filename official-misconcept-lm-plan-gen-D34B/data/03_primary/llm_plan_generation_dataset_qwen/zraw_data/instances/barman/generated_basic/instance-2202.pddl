(define (problem prob)
 (:domain barman)
 (:objects 
      shaker395 - shaker
      left right - hand
      shot114 shot286 shot330 - shot
      ingredient432 ingredient492 ingredient446 ingredient461 - ingredient
      cocktail71 - cocktail
      dispenser127 dispenser439 dispenser236 dispenser322 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker395)
  (ontable shot114)
  (ontable shot286)
  (ontable shot330)
  (dispenses dispenser127 ingredient432)
  (dispenses dispenser439 ingredient492)
  (dispenses dispenser236 ingredient446)
  (dispenses dispenser322 ingredient461)
  (clean shaker395)
  (clean shot114)
  (clean shot286)
  (clean shot330)
  (empty shaker395)
  (empty shot114)
  (empty shot286)
  (empty shot330)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker395 l0)
  (shaker-level shaker395 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail71 ingredient492)
  (cocktail-part2 cocktail71 ingredient446)
)
 (:goal
  (and
      (contains shot114 cocktail71)
      (contains shot286 ingredient492)
)))

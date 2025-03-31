(define (problem prob)
 (:domain barman)
 (:objects 
      shaker259 - shaker
      left right - hand
      shot328 shot383 - shot
      ingredient138 ingredient123 ingredient24 ingredient301 - ingredient
      cocktail1 - cocktail
      dispenser474 dispenser70 dispenser46 dispenser306 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker259)
  (ontable shot328)
  (ontable shot383)
  (dispenses dispenser474 ingredient138)
  (dispenses dispenser70 ingredient123)
  (dispenses dispenser46 ingredient24)
  (dispenses dispenser306 ingredient301)
  (clean shaker259)
  (clean shot328)
  (clean shot383)
  (empty shaker259)
  (empty shot328)
  (empty shot383)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker259 l0)
  (shaker-level shaker259 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient138)
  (cocktail-part2 cocktail1 ingredient24)
)
 (:goal
  (and
      (contains shot328 cocktail1)
)))

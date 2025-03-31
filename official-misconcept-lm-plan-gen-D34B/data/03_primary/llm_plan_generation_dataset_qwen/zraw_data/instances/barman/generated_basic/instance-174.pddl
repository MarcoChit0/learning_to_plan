(define (problem prob)
 (:domain barman)
 (:objects 
      shaker258 - shaker
      left right - hand
      shot308 shot226 - shot
      ingredient16 ingredient166 ingredient432 - ingredient
      cocktail1 - cocktail
      dispenser260 dispenser296 dispenser23 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker258)
  (ontable shot308)
  (ontable shot226)
  (dispenses dispenser260 ingredient16)
  (dispenses dispenser296 ingredient166)
  (dispenses dispenser23 ingredient432)
  (clean shaker258)
  (clean shot308)
  (clean shot226)
  (empty shaker258)
  (empty shot308)
  (empty shot226)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker258 l0)
  (shaker-level shaker258 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient16)
  (cocktail-part2 cocktail1 ingredient432)
)
 (:goal
  (and
      (contains shot308 cocktail1)
)))

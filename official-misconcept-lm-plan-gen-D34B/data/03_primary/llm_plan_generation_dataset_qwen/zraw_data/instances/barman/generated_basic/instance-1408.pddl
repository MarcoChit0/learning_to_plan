(define (problem prob)
 (:domain barman)
 (:objects 
      shaker232 - shaker
      left right - hand
      shot112 - shot
      ingredient151 ingredient306 ingredient359 ingredient16 - ingredient
      cocktail7 - cocktail
      dispenser272 dispenser494 dispenser38 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker232)
  (ontable shot112)
  (dispenses dispenser272 ingredient151)
  (dispenses dispenser494 ingredient306)
  (dispenses dispenser38 ingredient359)
  (dispenses dispenser146 ingredient16)
  (clean shaker232)
  (clean shot112)
  (empty shaker232)
  (empty shot112)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker232 l0)
  (shaker-level shaker232 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail7 ingredient151)
  (cocktail-part2 cocktail7 ingredient306)
)
 (:goal
  (and
      (contains shot112 cocktail7)
)))

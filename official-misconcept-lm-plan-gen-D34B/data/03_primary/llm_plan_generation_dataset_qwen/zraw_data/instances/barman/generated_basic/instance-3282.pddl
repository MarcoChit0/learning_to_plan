(define (problem prob)
 (:domain barman)
 (:objects 
      shaker354 - shaker
      left right - hand
      shot492 shot202 - shot
      ingredient103 ingredient94 ingredient33 - ingredient
      cocktail367 - cocktail
      dispenser250 dispenser245 dispenser71 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker354)
  (ontable shot492)
  (ontable shot202)
  (dispenses dispenser250 ingredient103)
  (dispenses dispenser245 ingredient94)
  (dispenses dispenser71 ingredient33)
  (clean shaker354)
  (clean shot492)
  (clean shot202)
  (empty shaker354)
  (empty shot492)
  (empty shot202)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker354 l0)
  (shaker-level shaker354 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail367 ingredient94)
  (cocktail-part2 cocktail367 ingredient33)
)
 (:goal
  (and
      (contains shot492 cocktail367)
)))

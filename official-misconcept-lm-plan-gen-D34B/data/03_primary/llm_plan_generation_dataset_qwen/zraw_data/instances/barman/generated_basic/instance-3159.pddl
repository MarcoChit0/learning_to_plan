(define (problem prob)
 (:domain barman)
 (:objects 
      shaker77 - shaker
      left right - hand
      shot86 shot341 shot390 - shot
      ingredient438 ingredient233 ingredient210 ingredient239 - ingredient
      cocktail398 - cocktail
      dispenser146 dispenser83 dispenser490 dispenser327 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker77)
  (ontable shot86)
  (ontable shot341)
  (ontable shot390)
  (dispenses dispenser146 ingredient438)
  (dispenses dispenser83 ingredient233)
  (dispenses dispenser490 ingredient210)
  (dispenses dispenser327 ingredient239)
  (clean shaker77)
  (clean shot86)
  (clean shot341)
  (clean shot390)
  (empty shaker77)
  (empty shot86)
  (empty shot341)
  (empty shot390)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker77 l0)
  (shaker-level shaker77 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail398 ingredient239)
  (cocktail-part2 cocktail398 ingredient438)
)
 (:goal
  (and
      (contains shot86 cocktail398)
      (contains shot341 ingredient233)
)))

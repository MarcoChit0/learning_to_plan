(define (problem prob)
 (:domain barman)
 (:objects 
      shaker15 - shaker
      left right - hand
      shot9 shot241 - shot
      ingredient135 ingredient115 - ingredient
      cocktail86 - cocktail
      dispenser114 dispenser47 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker15)
  (ontable shot9)
  (ontable shot241)
  (dispenses dispenser114 ingredient135)
  (dispenses dispenser47 ingredient115)
  (clean shaker15)
  (clean shot9)
  (clean shot241)
  (empty shaker15)
  (empty shot9)
  (empty shot241)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker15 l0)
  (shaker-level shaker15 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail86 ingredient135)
  (cocktail-part2 cocktail86 ingredient115)
)
 (:goal
  (and
      (contains shot9 cocktail86)
)))

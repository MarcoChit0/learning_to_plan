(define (problem prob)
 (:domain barman)
 (:objects 
      shaker113 - shaker
      left right - hand
      shot233 - shot
      ingredient191 ingredient471 ingredient99 ingredient143 - ingredient
      cocktail1 - cocktail
      dispenser143 dispenser17 dispenser160 dispenser22 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker113)
  (ontable shot233)
  (dispenses dispenser143 ingredient191)
  (dispenses dispenser17 ingredient471)
  (dispenses dispenser160 ingredient99)
  (dispenses dispenser22 ingredient143)
  (clean shaker113)
  (clean shot233)
  (empty shaker113)
  (empty shot233)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker113 l0)
  (shaker-level shaker113 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient143)
  (cocktail-part2 cocktail1 ingredient471)
)
 (:goal
  (and
      (contains shot233 cocktail1)
)))

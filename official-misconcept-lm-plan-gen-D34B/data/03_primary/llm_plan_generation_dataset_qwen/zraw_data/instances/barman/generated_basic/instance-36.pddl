(define (problem prob)
 (:domain barman)
 (:objects 
      shaker153 - shaker
      left right - hand
      shot37 shot281 shot292 - shot
      ingredient309 ingredient176 ingredient209 ingredient403 - ingredient
      cocktail1 - cocktail
      dispenser275 dispenser235 dispenser57 dispenser417 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker153)
  (ontable shot37)
  (ontable shot281)
  (ontable shot292)
  (dispenses dispenser275 ingredient309)
  (dispenses dispenser235 ingredient176)
  (dispenses dispenser57 ingredient209)
  (dispenses dispenser417 ingredient403)
  (clean shaker153)
  (clean shot37)
  (clean shot281)
  (clean shot292)
  (empty shaker153)
  (empty shot37)
  (empty shot281)
  (empty shot292)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker153 l0)
  (shaker-level shaker153 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient309)
  (cocktail-part2 cocktail1 ingredient403)
)
 (:goal
  (and
      (contains shot37 cocktail1)
      (contains shot281 cocktail1)
)))

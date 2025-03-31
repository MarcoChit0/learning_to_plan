(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot384 - shot
      ingredient77 ingredient319 ingredient220 ingredient199 - ingredient
      cocktail239 - cocktail
      dispenser145 dispenser98 dispenser463 dispenser126 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot384)
  (dispenses dispenser145 ingredient77)
  (dispenses dispenser98 ingredient319)
  (dispenses dispenser463 ingredient220)
  (dispenses dispenser126 ingredient199)
  (clean shaker446)
  (clean shot384)
  (empty shaker446)
  (empty shot384)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail239 ingredient319)
  (cocktail-part2 cocktail239 ingredient199)
)
 (:goal
  (and
      (contains shot384 cocktail239)
)))

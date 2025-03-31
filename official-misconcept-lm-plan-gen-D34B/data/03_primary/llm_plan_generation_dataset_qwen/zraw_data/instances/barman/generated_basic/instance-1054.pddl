(define (problem prob)
 (:domain barman)
 (:objects 
      shaker411 - shaker
      left right - hand
      shot117 shot401 shot65 - shot
      ingredient388 ingredient220 ingredient340 ingredient492 - ingredient
      cocktail1 - cocktail
      dispenser463 dispenser295 dispenser86 dispenser33 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker411)
  (ontable shot117)
  (ontable shot401)
  (ontable shot65)
  (dispenses dispenser463 ingredient388)
  (dispenses dispenser295 ingredient220)
  (dispenses dispenser86 ingredient340)
  (dispenses dispenser33 ingredient492)
  (clean shaker411)
  (clean shot117)
  (clean shot401)
  (clean shot65)
  (empty shaker411)
  (empty shot117)
  (empty shot401)
  (empty shot65)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker411 l0)
  (shaker-level shaker411 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient220)
  (cocktail-part2 cocktail1 ingredient388)
)
 (:goal
  (and
      (contains shot117 cocktail1)
      (contains shot401 cocktail1)
)))

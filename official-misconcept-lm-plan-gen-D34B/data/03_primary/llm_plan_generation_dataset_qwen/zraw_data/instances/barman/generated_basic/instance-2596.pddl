(define (problem prob)
 (:domain barman)
 (:objects 
      shaker307 - shaker
      left right - hand
      shot426 shot401 - shot
      ingredient92 ingredient8 ingredient135 - ingredient
      cocktail108 - cocktail
      dispenser210 dispenser71 dispenser127 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker307)
  (ontable shot426)
  (ontable shot401)
  (dispenses dispenser210 ingredient92)
  (dispenses dispenser71 ingredient8)
  (dispenses dispenser127 ingredient135)
  (clean shaker307)
  (clean shot426)
  (clean shot401)
  (empty shaker307)
  (empty shot426)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker307 l0)
  (shaker-level shaker307 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail108 ingredient92)
  (cocktail-part2 cocktail108 ingredient8)
)
 (:goal
  (and
      (contains shot426 cocktail108)
)))

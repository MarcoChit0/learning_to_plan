(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot150 shot81 - shot
      ingredient288 ingredient151 ingredient129 - ingredient
      cocktail190 - cocktail
      dispenser75 dispenser215 dispenser390 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot150)
  (ontable shot81)
  (dispenses dispenser75 ingredient288)
  (dispenses dispenser215 ingredient151)
  (dispenses dispenser390 ingredient129)
  (clean shaker109)
  (clean shot150)
  (clean shot81)
  (empty shaker109)
  (empty shot150)
  (empty shot81)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail190 ingredient288)
  (cocktail-part2 cocktail190 ingredient129)
)
 (:goal
  (and
      (contains shot150 cocktail190)
)))

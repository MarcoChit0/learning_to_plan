(define (problem prob)
 (:domain barman)
 (:objects 
      shaker296 - shaker
      left right - hand
      shot250 shot205 shot139 - shot
      ingredient463 ingredient489 ingredient107 ingredient323 - ingredient
      cocktail1 - cocktail
      dispenser113 dispenser416 dispenser197 dispenser7 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker296)
  (ontable shot250)
  (ontable shot205)
  (ontable shot139)
  (dispenses dispenser113 ingredient463)
  (dispenses dispenser416 ingredient489)
  (dispenses dispenser197 ingredient107)
  (dispenses dispenser7 ingredient323)
  (clean shaker296)
  (clean shot250)
  (clean shot205)
  (clean shot139)
  (empty shaker296)
  (empty shot250)
  (empty shot205)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker296 l0)
  (shaker-level shaker296 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient323)
  (cocktail-part2 cocktail1 ingredient463)
)
 (:goal
  (and
      (contains shot250 cocktail1)
      (contains shot205 cocktail1)
)))

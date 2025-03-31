(define (problem prob)
 (:domain barman)
 (:objects 
      shaker170 - shaker
      left right - hand
      shot390 shot419 shot254 shot312 - shot
      ingredient4 ingredient438 - ingredient
      cocktail203 - cocktail
      dispenser80 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker170)
  (ontable shot390)
  (ontable shot419)
  (ontable shot254)
  (ontable shot312)
  (dispenses dispenser80 ingredient4)
  (dispenses dispenser352 ingredient438)
  (clean shaker170)
  (clean shot390)
  (clean shot419)
  (clean shot254)
  (clean shot312)
  (empty shaker170)
  (empty shot390)
  (empty shot419)
  (empty shot254)
  (empty shot312)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker170 l0)
  (shaker-level shaker170 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail203 ingredient4)
  (cocktail-part2 cocktail203 ingredient438)
)
 (:goal
  (and
      (contains shot390 cocktail203)
      (contains shot419 cocktail203)
      (contains shot254 ingredient438)
)))

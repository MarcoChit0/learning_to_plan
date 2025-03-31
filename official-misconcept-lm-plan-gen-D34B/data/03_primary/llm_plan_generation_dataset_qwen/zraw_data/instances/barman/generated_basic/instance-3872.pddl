(define (problem prob)
 (:domain barman)
 (:objects 
      shaker360 - shaker
      left right - hand
      shot185 shot324 - shot
      ingredient155 ingredient210 ingredient200 - ingredient
      cocktail240 - cocktail
      dispenser90 dispenser196 dispenser474 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker360)
  (ontable shot185)
  (ontable shot324)
  (dispenses dispenser90 ingredient155)
  (dispenses dispenser196 ingredient210)
  (dispenses dispenser474 ingredient200)
  (clean shaker360)
  (clean shot185)
  (clean shot324)
  (empty shaker360)
  (empty shot185)
  (empty shot324)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker360 l0)
  (shaker-level shaker360 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail240 ingredient210)
  (cocktail-part2 cocktail240 ingredient200)
)
 (:goal
  (and
      (contains shot185 cocktail240)
)))

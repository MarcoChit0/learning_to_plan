(define (problem prob)
 (:domain barman)
 (:objects 
      shaker332 - shaker
      left right - hand
      shot21 shot98 shot62 - shot
      ingredient155 ingredient59 ingredient131 - ingredient
      cocktail330 - cocktail
      dispenser421 dispenser311 dispenser215 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker332)
  (ontable shot21)
  (ontable shot98)
  (ontable shot62)
  (dispenses dispenser421 ingredient155)
  (dispenses dispenser311 ingredient59)
  (dispenses dispenser215 ingredient131)
  (clean shaker332)
  (clean shot21)
  (clean shot98)
  (clean shot62)
  (empty shaker332)
  (empty shot21)
  (empty shot98)
  (empty shot62)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker332 l0)
  (shaker-level shaker332 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail330 ingredient131)
  (cocktail-part2 cocktail330 ingredient59)
)
 (:goal
  (and
      (contains shot21 cocktail330)
      (contains shot98 cocktail330)
)))

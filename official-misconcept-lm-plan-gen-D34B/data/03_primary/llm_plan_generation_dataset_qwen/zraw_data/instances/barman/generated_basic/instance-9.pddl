(define (problem prob)
 (:domain barman)
 (:objects 
      shaker7 - shaker
      left right - hand
      shot212 shot230 - shot
      ingredient40 ingredient359 - ingredient
      cocktail1 - cocktail
      dispenser57 dispenser447 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker7)
  (ontable shot212)
  (ontable shot230)
  (dispenses dispenser57 ingredient40)
  (dispenses dispenser447 ingredient359)
  (clean shaker7)
  (clean shot212)
  (clean shot230)
  (empty shaker7)
  (empty shot212)
  (empty shot230)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker7 l0)
  (shaker-level shaker7 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient359)
  (cocktail-part2 cocktail1 ingredient40)
)
 (:goal
  (and
      (contains shot212 cocktail1)
)))

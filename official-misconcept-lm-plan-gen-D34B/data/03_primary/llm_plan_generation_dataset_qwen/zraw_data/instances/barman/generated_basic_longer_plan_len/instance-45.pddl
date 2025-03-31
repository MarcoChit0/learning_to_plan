(define (problem prob)
 (:domain barman)
 (:objects 
      shaker468 - shaker
      left right - hand
      shot306 shot200 shot14 - shot
      ingredient306 ingredient459 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser87 dispenser324 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker468)
  (ontable shot306)
  (ontable shot200)
  (ontable shot14)
  (dispenses dispenser87 ingredient306)
  (dispenses dispenser324 ingredient459)
  (clean shaker468)
  (clean shot306)
  (clean shot200)
  (clean shot14)
  (empty shaker468)
  (empty shot306)
  (empty shot200)
  (empty shot14)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker468 l0)
  (shaker-level shaker468 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient306)
  (cocktail-part2 cocktail1 ingredient459)
  (cocktail-part1 cocktail2 ingredient306)
  (cocktail-part2 cocktail2 ingredient459)
)
 (:goal
  (and
      (contains shot306 cocktail1)
      (contains shot200 cocktail2)
)))

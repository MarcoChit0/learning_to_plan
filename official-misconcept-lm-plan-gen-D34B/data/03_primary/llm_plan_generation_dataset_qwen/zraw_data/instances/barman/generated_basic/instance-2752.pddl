(define (problem prob)
 (:domain barman)
 (:objects 
      shaker426 - shaker
      left right - hand
      shot56 - shot
      ingredient289 ingredient368 - ingredient
      cocktail49 - cocktail
      dispenser305 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker426)
  (ontable shot56)
  (dispenses dispenser305 ingredient289)
  (dispenses dispenser271 ingredient368)
  (clean shaker426)
  (clean shot56)
  (empty shaker426)
  (empty shot56)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker426 l0)
  (shaker-level shaker426 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail49 ingredient368)
  (cocktail-part2 cocktail49 ingredient289)
)
 (:goal
  (and
      (contains shot56 cocktail49)
)))

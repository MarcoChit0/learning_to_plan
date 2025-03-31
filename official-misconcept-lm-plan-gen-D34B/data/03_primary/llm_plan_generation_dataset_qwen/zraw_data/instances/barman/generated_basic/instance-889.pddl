(define (problem prob)
 (:domain barman)
 (:objects 
      shaker402 - shaker
      left right - hand
      shot25 shot35 - shot
      ingredient101 ingredient361 - ingredient
      cocktail1 - cocktail
      dispenser418 dispenser65 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker402)
  (ontable shot25)
  (ontable shot35)
  (dispenses dispenser418 ingredient101)
  (dispenses dispenser65 ingredient361)
  (clean shaker402)
  (clean shot25)
  (clean shot35)
  (empty shaker402)
  (empty shot25)
  (empty shot35)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker402 l0)
  (shaker-level shaker402 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient101)
  (cocktail-part2 cocktail1 ingredient361)
)
 (:goal
  (and
      (contains shot25 cocktail1)
)))

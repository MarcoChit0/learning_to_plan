(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot437 shot47 - shot
      ingredient402 ingredient55 - ingredient
      cocktail1 - cocktail
      dispenser175 dispenser461 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot437)
  (ontable shot47)
  (dispenses dispenser175 ingredient402)
  (dispenses dispenser461 ingredient55)
  (clean shaker386)
  (clean shot437)
  (clean shot47)
  (empty shaker386)
  (empty shot437)
  (empty shot47)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient55)
  (cocktail-part2 cocktail1 ingredient402)
)
 (:goal
  (and
      (contains shot437 cocktail1)
)))

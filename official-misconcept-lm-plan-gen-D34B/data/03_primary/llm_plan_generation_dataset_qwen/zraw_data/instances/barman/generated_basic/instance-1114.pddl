(define (problem prob)
 (:domain barman)
 (:objects 
      shaker306 - shaker
      left right - hand
      shot127 shot483 - shot
      ingredient55 ingredient129 - ingredient
      cocktail1 - cocktail
      dispenser9 dispenser207 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker306)
  (ontable shot127)
  (ontable shot483)
  (dispenses dispenser9 ingredient55)
  (dispenses dispenser207 ingredient129)
  (clean shaker306)
  (clean shot127)
  (clean shot483)
  (empty shaker306)
  (empty shot127)
  (empty shot483)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker306 l0)
  (shaker-level shaker306 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient55)
  (cocktail-part2 cocktail1 ingredient129)
)
 (:goal
  (and
      (contains shot127 cocktail1)
)))

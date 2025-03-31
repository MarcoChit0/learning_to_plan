(define (problem prob)
 (:domain barman)
 (:objects 
      shaker16 - shaker
      left right - hand
      shot86 - shot
      ingredient38 ingredient78 ingredient366 ingredient47 - ingredient
      cocktail447 - cocktail
      dispenser457 dispenser62 dispenser477 dispenser66 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker16)
  (ontable shot86)
  (dispenses dispenser457 ingredient38)
  (dispenses dispenser62 ingredient78)
  (dispenses dispenser477 ingredient366)
  (dispenses dispenser66 ingredient47)
  (clean shaker16)
  (clean shot86)
  (empty shaker16)
  (empty shot86)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker16 l0)
  (shaker-level shaker16 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail447 ingredient366)
  (cocktail-part2 cocktail447 ingredient38)
)
 (:goal
  (and
      (contains shot86 cocktail447)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker107 - shaker
      left right - hand
      shot389 shot300 - shot
      ingredient421 ingredient282 ingredient124 ingredient317 - ingredient
      cocktail1 - cocktail
      dispenser191 dispenser333 dispenser52 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker107)
  (ontable shot389)
  (ontable shot300)
  (dispenses dispenser191 ingredient421)
  (dispenses dispenser333 ingredient282)
  (dispenses dispenser52 ingredient124)
  (dispenses dispenser62 ingredient317)
  (clean shaker107)
  (clean shot389)
  (clean shot300)
  (empty shaker107)
  (empty shot389)
  (empty shot300)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker107 l0)
  (shaker-level shaker107 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient124)
  (cocktail-part2 cocktail1 ingredient421)
)
 (:goal
  (and
      (contains shot389 cocktail1)
)))

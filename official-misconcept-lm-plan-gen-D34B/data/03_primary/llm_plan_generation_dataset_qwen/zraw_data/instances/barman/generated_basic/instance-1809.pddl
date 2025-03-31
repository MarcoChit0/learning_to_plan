(define (problem prob)
 (:domain barman)
 (:objects 
      shaker321 - shaker
      left right - hand
      shot421 shot480 - shot
      ingredient361 ingredient277 ingredient293 ingredient247 - ingredient
      cocktail123 - cocktail
      dispenser247 dispenser380 dispenser431 dispenser20 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker321)
  (ontable shot421)
  (ontable shot480)
  (dispenses dispenser247 ingredient361)
  (dispenses dispenser380 ingredient277)
  (dispenses dispenser431 ingredient293)
  (dispenses dispenser20 ingredient247)
  (clean shaker321)
  (clean shot421)
  (clean shot480)
  (empty shaker321)
  (empty shot421)
  (empty shot480)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker321 l0)
  (shaker-level shaker321 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail123 ingredient361)
  (cocktail-part2 cocktail123 ingredient247)
)
 (:goal
  (and
      (contains shot421 cocktail123)
)))

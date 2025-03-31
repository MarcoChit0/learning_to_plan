(define (problem prob)
 (:domain barman)
 (:objects 
      shaker386 - shaker
      left right - hand
      shot442 shot479 - shot
      ingredient11 ingredient315 ingredient463 ingredient308 - ingredient
      cocktail1 - cocktail
      dispenser180 dispenser333 dispenser89 dispenser62 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker386)
  (ontable shot442)
  (ontable shot479)
  (dispenses dispenser180 ingredient11)
  (dispenses dispenser333 ingredient315)
  (dispenses dispenser89 ingredient463)
  (dispenses dispenser62 ingredient308)
  (clean shaker386)
  (clean shot442)
  (clean shot479)
  (empty shaker386)
  (empty shot442)
  (empty shot479)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker386 l0)
  (shaker-level shaker386 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient315)
  (cocktail-part2 cocktail1 ingredient308)
)
 (:goal
  (and
      (contains shot442 cocktail1)
)))

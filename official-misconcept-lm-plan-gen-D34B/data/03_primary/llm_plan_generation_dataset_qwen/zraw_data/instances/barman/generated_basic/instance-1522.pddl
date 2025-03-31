(define (problem prob)
 (:domain barman)
 (:objects 
      shaker240 - shaker
      left right - hand
      shot365 shot129 - shot
      ingredient242 ingredient311 ingredient468 ingredient161 - ingredient
      cocktail319 - cocktail
      dispenser67 dispenser366 dispenser158 dispenser446 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker240)
  (ontable shot365)
  (ontable shot129)
  (dispenses dispenser67 ingredient242)
  (dispenses dispenser366 ingredient311)
  (dispenses dispenser158 ingredient468)
  (dispenses dispenser446 ingredient161)
  (clean shaker240)
  (clean shot365)
  (clean shot129)
  (empty shaker240)
  (empty shot365)
  (empty shot129)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker240 l0)
  (shaker-level shaker240 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail319 ingredient468)
  (cocktail-part2 cocktail319 ingredient311)
)
 (:goal
  (and
      (contains shot365 cocktail319)
)))

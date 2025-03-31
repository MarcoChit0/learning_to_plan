(define (problem prob)
 (:domain barman)
 (:objects 
      shaker358 - shaker
      left right - hand
      shot232 shot185 - shot
      ingredient189 ingredient24 ingredient319 - ingredient
      cocktail6 - cocktail
      dispenser356 dispenser9 dispenser90 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker358)
  (ontable shot232)
  (ontable shot185)
  (dispenses dispenser356 ingredient189)
  (dispenses dispenser9 ingredient24)
  (dispenses dispenser90 ingredient319)
  (clean shaker358)
  (clean shot232)
  (clean shot185)
  (empty shaker358)
  (empty shot232)
  (empty shot185)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker358 l0)
  (shaker-level shaker358 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient24)
  (cocktail-part2 cocktail6 ingredient319)
)
 (:goal
  (and
      (contains shot232 cocktail6)
)))

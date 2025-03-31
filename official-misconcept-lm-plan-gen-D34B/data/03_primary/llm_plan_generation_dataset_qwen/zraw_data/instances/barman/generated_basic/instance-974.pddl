(define (problem prob)
 (:domain barman)
 (:objects 
      shaker341 - shaker
      left right - hand
      shot270 shot93 - shot
      ingredient284 ingredient265 ingredient482 ingredient261 - ingredient
      cocktail1 - cocktail
      dispenser247 dispenser321 dispenser333 dispenser163 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker341)
  (ontable shot270)
  (ontable shot93)
  (dispenses dispenser247 ingredient284)
  (dispenses dispenser321 ingredient265)
  (dispenses dispenser333 ingredient482)
  (dispenses dispenser163 ingredient261)
  (clean shaker341)
  (clean shot270)
  (clean shot93)
  (empty shaker341)
  (empty shot270)
  (empty shot93)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker341 l0)
  (shaker-level shaker341 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient265)
  (cocktail-part2 cocktail1 ingredient261)
)
 (:goal
  (and
      (contains shot270 cocktail1)
)))

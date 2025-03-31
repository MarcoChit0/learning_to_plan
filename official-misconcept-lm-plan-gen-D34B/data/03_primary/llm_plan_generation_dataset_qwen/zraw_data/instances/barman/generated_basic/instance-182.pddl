(define (problem prob)
 (:domain barman)
 (:objects 
      shaker476 - shaker
      left right - hand
      shot89 shot212 shot331 - shot
      ingredient348 ingredient146 ingredient138 ingredient431 - ingredient
      cocktail1 - cocktail
      dispenser145 dispenser215 dispenser417 dispenser259 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker476)
  (ontable shot89)
  (ontable shot212)
  (ontable shot331)
  (dispenses dispenser145 ingredient348)
  (dispenses dispenser215 ingredient146)
  (dispenses dispenser417 ingredient138)
  (dispenses dispenser259 ingredient431)
  (clean shaker476)
  (clean shot89)
  (clean shot212)
  (clean shot331)
  (empty shaker476)
  (empty shot89)
  (empty shot212)
  (empty shot331)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker476 l0)
  (shaker-level shaker476 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient431)
  (cocktail-part2 cocktail1 ingredient138)
)
 (:goal
  (and
      (contains shot89 cocktail1)
      (contains shot212 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker263 - shaker
      left right - hand
      shot163 shot0 - shot
      ingredient238 ingredient27 ingredient319 ingredient284 - ingredient
      cocktail243 - cocktail
      dispenser234 dispenser149 dispenser221 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker263)
  (ontable shot163)
  (ontable shot0)
  (dispenses dispenser234 ingredient238)
  (dispenses dispenser149 ingredient27)
  (dispenses dispenser221 ingredient319)
  (dispenses dispenser69 ingredient284)
  (clean shaker263)
  (clean shot163)
  (clean shot0)
  (empty shaker263)
  (empty shot163)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker263 l0)
  (shaker-level shaker263 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail243 ingredient27)
  (cocktail-part2 cocktail243 ingredient319)
)
 (:goal
  (and
      (contains shot163 cocktail243)
)))

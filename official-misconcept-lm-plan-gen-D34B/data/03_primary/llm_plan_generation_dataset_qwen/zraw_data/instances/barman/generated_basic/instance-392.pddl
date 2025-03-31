(define (problem prob)
 (:domain barman)
 (:objects 
      shaker422 - shaker
      left right - hand
      shot467 shot368 shot437 - shot
      ingredient83 ingredient211 ingredient14 - ingredient
      cocktail1 - cocktail
      dispenser437 dispenser487 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker422)
  (ontable shot467)
  (ontable shot368)
  (ontable shot437)
  (dispenses dispenser437 ingredient83)
  (dispenses dispenser487 ingredient211)
  (dispenses dispenser333 ingredient14)
  (clean shaker422)
  (clean shot467)
  (clean shot368)
  (clean shot437)
  (empty shaker422)
  (empty shot467)
  (empty shot368)
  (empty shot437)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker422 l0)
  (shaker-level shaker422 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient14)
  (cocktail-part2 cocktail1 ingredient83)
)
 (:goal
  (and
      (contains shot467 cocktail1)
      (contains shot368 ingredient83)
)))

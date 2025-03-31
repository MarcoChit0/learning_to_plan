(define (problem prob)
 (:domain barman)
 (:objects 
      shaker109 - shaker
      left right - hand
      shot301 shot124 - shot
      ingredient238 ingredient64 - ingredient
      cocktail328 - cocktail
      dispenser403 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker109)
  (ontable shot301)
  (ontable shot124)
  (dispenses dispenser403 ingredient238)
  (dispenses dispenser49 ingredient64)
  (clean shaker109)
  (clean shot301)
  (clean shot124)
  (empty shaker109)
  (empty shot301)
  (empty shot124)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker109 l0)
  (shaker-level shaker109 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail328 ingredient64)
  (cocktail-part2 cocktail328 ingredient238)
)
 (:goal
  (and
      (contains shot301 cocktail328)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker4 - shaker
      left right - hand
      shot422 shot396 - shot
      ingredient346 ingredient476 - ingredient
      cocktail1 - cocktail
      dispenser398 dispenser357 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker4)
  (ontable shot422)
  (ontable shot396)
  (dispenses dispenser398 ingredient346)
  (dispenses dispenser357 ingredient476)
  (clean shaker4)
  (clean shot422)
  (clean shot396)
  (empty shaker4)
  (empty shot422)
  (empty shot396)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker4 l0)
  (shaker-level shaker4 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient346)
  (cocktail-part2 cocktail1 ingredient476)
)
 (:goal
  (and
      (contains shot422 cocktail1)
)))

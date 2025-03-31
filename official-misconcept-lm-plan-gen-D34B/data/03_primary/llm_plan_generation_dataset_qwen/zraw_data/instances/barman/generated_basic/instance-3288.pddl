(define (problem prob)
 (:domain barman)
 (:objects 
      shaker405 - shaker
      left right - hand
      shot219 shot353 - shot
      ingredient312 ingredient24 ingredient188 - ingredient
      cocktail107 - cocktail
      dispenser72 dispenser493 dispenser335 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker405)
  (ontable shot219)
  (ontable shot353)
  (dispenses dispenser72 ingredient312)
  (dispenses dispenser493 ingredient24)
  (dispenses dispenser335 ingredient188)
  (clean shaker405)
  (clean shot219)
  (clean shot353)
  (empty shaker405)
  (empty shot219)
  (empty shot353)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker405 l0)
  (shaker-level shaker405 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail107 ingredient24)
  (cocktail-part2 cocktail107 ingredient312)
)
 (:goal
  (and
      (contains shot219 cocktail107)
)))

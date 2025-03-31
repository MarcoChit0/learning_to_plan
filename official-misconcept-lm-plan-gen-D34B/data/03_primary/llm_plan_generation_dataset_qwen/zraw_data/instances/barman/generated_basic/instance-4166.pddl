(define (problem prob)
 (:domain barman)
 (:objects 
      shaker281 - shaker
      left right - hand
      shot48 shot401 - shot
      ingredient220 ingredient119 - ingredient
      cocktail252 - cocktail
      dispenser456 dispenser393 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker281)
  (ontable shot48)
  (ontable shot401)
  (dispenses dispenser456 ingredient220)
  (dispenses dispenser393 ingredient119)
  (clean shaker281)
  (clean shot48)
  (clean shot401)
  (empty shaker281)
  (empty shot48)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker281 l0)
  (shaker-level shaker281 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail252 ingredient119)
  (cocktail-part2 cocktail252 ingredient220)
)
 (:goal
  (and
      (contains shot48 cocktail252)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker57 - shaker
      left right - hand
      shot401 - shot
      ingredient281 ingredient59 ingredient246 - ingredient
      cocktail38 - cocktail
      dispenser443 dispenser13 dispenser352 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker57)
  (ontable shot401)
  (dispenses dispenser443 ingredient281)
  (dispenses dispenser13 ingredient59)
  (dispenses dispenser352 ingredient246)
  (clean shaker57)
  (clean shot401)
  (empty shaker57)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker57 l0)
  (shaker-level shaker57 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail38 ingredient59)
  (cocktail-part2 cocktail38 ingredient281)
)
 (:goal
  (and
      (contains shot401 cocktail38)
)))

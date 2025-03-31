(define (problem prob)
 (:domain barman)
 (:objects 
      shaker234 - shaker
      left right - hand
      shot23 shot219 shot155 - shot
      ingredient281 ingredient420 - ingredient
      cocktail351 - cocktail
      dispenser483 dispenser378 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker234)
  (ontable shot23)
  (ontable shot219)
  (ontable shot155)
  (dispenses dispenser483 ingredient281)
  (dispenses dispenser378 ingredient420)
  (clean shaker234)
  (clean shot23)
  (clean shot219)
  (clean shot155)
  (empty shaker234)
  (empty shot23)
  (empty shot219)
  (empty shot155)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker234 l0)
  (shaker-level shaker234 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail351 ingredient420)
  (cocktail-part2 cocktail351 ingredient281)
)
 (:goal
  (and
      (contains shot23 cocktail351)
      (contains shot219 ingredient281)
)))

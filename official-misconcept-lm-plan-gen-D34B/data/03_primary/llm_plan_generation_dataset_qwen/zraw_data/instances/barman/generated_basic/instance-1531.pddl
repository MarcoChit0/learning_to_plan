(define (problem prob)
 (:domain barman)
 (:objects 
      shaker221 - shaker
      left right - hand
      shot442 shot281 - shot
      ingredient465 ingredient360 ingredient212 ingredient125 - ingredient
      cocktail219 - cocktail
      dispenser119 dispenser355 dispenser430 dispenser384 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker221)
  (ontable shot442)
  (ontable shot281)
  (dispenses dispenser119 ingredient465)
  (dispenses dispenser355 ingredient360)
  (dispenses dispenser430 ingredient212)
  (dispenses dispenser384 ingredient125)
  (clean shaker221)
  (clean shot442)
  (clean shot281)
  (empty shaker221)
  (empty shot442)
  (empty shot281)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker221 l0)
  (shaker-level shaker221 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail219 ingredient212)
  (cocktail-part2 cocktail219 ingredient125)
)
 (:goal
  (and
      (contains shot442 cocktail219)
)))

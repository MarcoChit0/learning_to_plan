(define (problem prob)
 (:domain barman)
 (:objects 
      shaker257 - shaker
      left right - hand
      shot349 - shot
      ingredient344 ingredient499 ingredient466 ingredient179 - ingredient
      cocktail120 - cocktail
      dispenser125 dispenser274 dispenser71 dispenser246 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker257)
  (ontable shot349)
  (dispenses dispenser125 ingredient344)
  (dispenses dispenser274 ingredient499)
  (dispenses dispenser71 ingredient466)
  (dispenses dispenser246 ingredient179)
  (clean shaker257)
  (clean shot349)
  (empty shaker257)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker257 l0)
  (shaker-level shaker257 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail120 ingredient344)
  (cocktail-part2 cocktail120 ingredient466)
)
 (:goal
  (and
      (contains shot349 cocktail120)
)))

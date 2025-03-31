(define (problem prob)
 (:domain barman)
 (:objects 
      shaker370 - shaker
      left right - hand
      shot248 shot225 shot72 - shot
      ingredient96 ingredient474 ingredient372 ingredient120 - ingredient
      cocktail464 - cocktail
      dispenser197 dispenser189 dispenser206 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker370)
  (ontable shot248)
  (ontable shot225)
  (ontable shot72)
  (dispenses dispenser197 ingredient96)
  (dispenses dispenser189 ingredient474)
  (dispenses dispenser206 ingredient372)
  (dispenses dispenser266 ingredient120)
  (clean shaker370)
  (clean shot248)
  (clean shot225)
  (clean shot72)
  (empty shaker370)
  (empty shot248)
  (empty shot225)
  (empty shot72)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker370 l0)
  (shaker-level shaker370 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail464 ingredient372)
  (cocktail-part2 cocktail464 ingredient96)
)
 (:goal
  (and
      (contains shot248 cocktail464)
      (contains shot225 ingredient96)
)))

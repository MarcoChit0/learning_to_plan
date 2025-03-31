(define (problem prob)
 (:domain barman)
 (:objects 
      shaker91 - shaker
      left right - hand
      shot255 - shot
      ingredient285 ingredient320 ingredient482 ingredient104 - ingredient
      cocktail469 - cocktail
      dispenser481 dispenser422 dispenser326 dispenser337 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker91)
  (ontable shot255)
  (dispenses dispenser481 ingredient285)
  (dispenses dispenser422 ingredient320)
  (dispenses dispenser326 ingredient482)
  (dispenses dispenser337 ingredient104)
  (clean shaker91)
  (clean shot255)
  (empty shaker91)
  (empty shot255)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker91 l0)
  (shaker-level shaker91 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail469 ingredient320)
  (cocktail-part2 cocktail469 ingredient482)
)
 (:goal
  (and
      (contains shot255 cocktail469)
)))

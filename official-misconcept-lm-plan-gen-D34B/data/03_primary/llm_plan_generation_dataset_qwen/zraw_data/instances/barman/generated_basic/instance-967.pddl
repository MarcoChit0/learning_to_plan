(define (problem prob)
 (:domain barman)
 (:objects 
      shaker270 - shaker
      left right - hand
      shot242 - shot
      ingredient356 ingredient110 ingredient281 ingredient79 - ingredient
      cocktail1 - cocktail
      dispenser340 dispenser54 dispenser70 dispenser422 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker270)
  (ontable shot242)
  (dispenses dispenser340 ingredient356)
  (dispenses dispenser54 ingredient110)
  (dispenses dispenser70 ingredient281)
  (dispenses dispenser422 ingredient79)
  (clean shaker270)
  (clean shot242)
  (empty shaker270)
  (empty shot242)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker270 l0)
  (shaker-level shaker270 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient356)
  (cocktail-part2 cocktail1 ingredient79)
)
 (:goal
  (and
      (contains shot242 cocktail1)
)))

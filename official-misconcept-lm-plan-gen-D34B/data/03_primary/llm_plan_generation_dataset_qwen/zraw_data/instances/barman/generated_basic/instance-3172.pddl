(define (problem prob)
 (:domain barman)
 (:objects 
      shaker362 - shaker
      left right - hand
      shot452 - shot
      ingredient438 ingredient342 ingredient281 - ingredient
      cocktail268 - cocktail
      dispenser32 dispenser189 dispenser403 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker362)
  (ontable shot452)
  (dispenses dispenser32 ingredient438)
  (dispenses dispenser189 ingredient342)
  (dispenses dispenser403 ingredient281)
  (clean shaker362)
  (clean shot452)
  (empty shaker362)
  (empty shot452)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker362 l0)
  (shaker-level shaker362 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail268 ingredient438)
  (cocktail-part2 cocktail268 ingredient281)
)
 (:goal
  (and
      (contains shot452 cocktail268)
)))

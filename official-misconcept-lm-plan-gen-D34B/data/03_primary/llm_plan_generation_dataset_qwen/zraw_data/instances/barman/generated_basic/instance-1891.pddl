(define (problem prob)
 (:domain barman)
 (:objects 
      shaker17 - shaker
      left right - hand
      shot58 shot298 - shot
      ingredient201 ingredient270 ingredient428 - ingredient
      cocktail6 - cocktail
      dispenser304 dispenser288 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker17)
  (ontable shot58)
  (ontable shot298)
  (dispenses dispenser304 ingredient201)
  (dispenses dispenser288 ingredient270)
  (dispenses dispenser387 ingredient428)
  (clean shaker17)
  (clean shot58)
  (clean shot298)
  (empty shaker17)
  (empty shot58)
  (empty shot298)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker17 l0)
  (shaker-level shaker17 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail6 ingredient270)
  (cocktail-part2 cocktail6 ingredient201)
)
 (:goal
  (and
      (contains shot58 cocktail6)
)))

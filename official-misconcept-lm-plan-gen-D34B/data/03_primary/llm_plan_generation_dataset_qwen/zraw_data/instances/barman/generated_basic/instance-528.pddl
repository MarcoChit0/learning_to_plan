(define (problem prob)
 (:domain barman)
 (:objects 
      shaker428 - shaker
      left right - hand
      shot304 shot456 - shot
      ingredient295 ingredient24 - ingredient
      cocktail1 - cocktail
      dispenser336 dispenser362 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker428)
  (ontable shot304)
  (ontable shot456)
  (dispenses dispenser336 ingredient295)
  (dispenses dispenser362 ingredient24)
  (clean shaker428)
  (clean shot304)
  (clean shot456)
  (empty shaker428)
  (empty shot304)
  (empty shot456)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker428 l0)
  (shaker-level shaker428 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient24)
  (cocktail-part2 cocktail1 ingredient295)
)
 (:goal
  (and
      (contains shot304 cocktail1)
)))

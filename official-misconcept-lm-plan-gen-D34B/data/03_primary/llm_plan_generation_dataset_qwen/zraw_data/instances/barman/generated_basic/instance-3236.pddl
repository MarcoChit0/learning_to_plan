(define (problem prob)
 (:domain barman)
 (:objects 
      shaker398 - shaker
      left right - hand
      shot499 - shot
      ingredient135 ingredient356 ingredient276 - ingredient
      cocktail242 - cocktail
      dispenser209 dispenser368 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker398)
  (ontable shot499)
  (dispenses dispenser209 ingredient135)
  (dispenses dispenser368 ingredient356)
  (dispenses dispenser161 ingredient276)
  (clean shaker398)
  (clean shot499)
  (empty shaker398)
  (empty shot499)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker398 l0)
  (shaker-level shaker398 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail242 ingredient276)
  (cocktail-part2 cocktail242 ingredient356)
)
 (:goal
  (and
      (contains shot499 cocktail242)
)))

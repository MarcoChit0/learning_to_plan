(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot177 shot484 - shot
      ingredient135 ingredient201 - ingredient
      cocktail55 - cocktail
      dispenser83 dispenser459 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot177)
  (ontable shot484)
  (dispenses dispenser83 ingredient135)
  (dispenses dispenser459 ingredient201)
  (clean shaker250)
  (clean shot177)
  (clean shot484)
  (empty shaker250)
  (empty shot177)
  (empty shot484)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail55 ingredient135)
  (cocktail-part2 cocktail55 ingredient201)
)
 (:goal
  (and
      (contains shot177 cocktail55)
)))

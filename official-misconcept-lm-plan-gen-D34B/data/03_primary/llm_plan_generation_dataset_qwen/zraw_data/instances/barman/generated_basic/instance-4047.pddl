(define (problem prob)
 (:domain barman)
 (:objects 
      shaker214 - shaker
      left right - hand
      shot51 shot387 - shot
      ingredient342 ingredient269 ingredient427 ingredient59 - ingredient
      cocktail227 - cocktail
      dispenser96 dispenser214 dispenser300 dispenser69 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker214)
  (ontable shot51)
  (ontable shot387)
  (dispenses dispenser96 ingredient342)
  (dispenses dispenser214 ingredient269)
  (dispenses dispenser300 ingredient427)
  (dispenses dispenser69 ingredient59)
  (clean shaker214)
  (clean shot51)
  (clean shot387)
  (empty shaker214)
  (empty shot51)
  (empty shot387)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker214 l0)
  (shaker-level shaker214 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail227 ingredient342)
  (cocktail-part2 cocktail227 ingredient427)
)
 (:goal
  (and
      (contains shot51 cocktail227)
)))

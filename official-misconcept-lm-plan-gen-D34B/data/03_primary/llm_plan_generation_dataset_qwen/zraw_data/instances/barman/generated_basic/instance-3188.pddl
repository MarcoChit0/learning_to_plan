(define (problem prob)
 (:domain barman)
 (:objects 
      shaker475 - shaker
      left right - hand
      shot453 shot77 - shot
      ingredient4 ingredient316 ingredient79 - ingredient
      cocktail414 - cocktail
      dispenser280 dispenser148 dispenser372 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker475)
  (ontable shot453)
  (ontable shot77)
  (dispenses dispenser280 ingredient4)
  (dispenses dispenser148 ingredient316)
  (dispenses dispenser372 ingredient79)
  (clean shaker475)
  (clean shot453)
  (clean shot77)
  (empty shaker475)
  (empty shot453)
  (empty shot77)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker475 l0)
  (shaker-level shaker475 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail414 ingredient4)
  (cocktail-part2 cocktail414 ingredient316)
)
 (:goal
  (and
      (contains shot453 cocktail414)
)))

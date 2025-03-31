(define (problem prob)
 (:domain barman)
 (:objects 
      shaker406 - shaker
      left right - hand
      shot496 shot175 - shot
      ingredient383 ingredient141 ingredient109 ingredient455 - ingredient
      cocktail1 - cocktail
      dispenser5 dispenser61 dispenser262 dispenser188 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker406)
  (ontable shot496)
  (ontable shot175)
  (dispenses dispenser5 ingredient383)
  (dispenses dispenser61 ingredient141)
  (dispenses dispenser262 ingredient109)
  (dispenses dispenser188 ingredient455)
  (clean shaker406)
  (clean shot496)
  (clean shot175)
  (empty shaker406)
  (empty shot496)
  (empty shot175)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker406 l0)
  (shaker-level shaker406 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient455)
  (cocktail-part2 cocktail1 ingredient383)
)
 (:goal
  (and
      (contains shot496 cocktail1)
)))

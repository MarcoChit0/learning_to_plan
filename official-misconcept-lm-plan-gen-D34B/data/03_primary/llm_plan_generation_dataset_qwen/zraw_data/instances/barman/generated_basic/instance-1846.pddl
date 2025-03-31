(define (problem prob)
 (:domain barman)
 (:objects 
      shaker171 - shaker
      left right - hand
      shot77 shot38 - shot
      ingredient411 ingredient9 ingredient494 - ingredient
      cocktail1 - cocktail
      dispenser13 dispenser467 dispenser358 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker171)
  (ontable shot77)
  (ontable shot38)
  (dispenses dispenser13 ingredient411)
  (dispenses dispenser467 ingredient9)
  (dispenses dispenser358 ingredient494)
  (clean shaker171)
  (clean shot77)
  (clean shot38)
  (empty shaker171)
  (empty shot77)
  (empty shot38)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker171 l0)
  (shaker-level shaker171 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient411)
  (cocktail-part2 cocktail1 ingredient9)
)
 (:goal
  (and
      (contains shot77 cocktail1)
)))

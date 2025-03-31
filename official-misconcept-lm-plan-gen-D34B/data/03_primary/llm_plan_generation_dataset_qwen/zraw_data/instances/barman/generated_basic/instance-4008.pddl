(define (problem prob)
 (:domain barman)
 (:objects 
      shaker175 - shaker
      left right - hand
      shot456 shot54 shot139 - shot
      ingredient474 ingredient101 - ingredient
      cocktail320 - cocktail
      dispenser263 dispenser191 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker175)
  (ontable shot456)
  (ontable shot54)
  (ontable shot139)
  (dispenses dispenser263 ingredient474)
  (dispenses dispenser191 ingredient101)
  (clean shaker175)
  (clean shot456)
  (clean shot54)
  (clean shot139)
  (empty shaker175)
  (empty shot456)
  (empty shot54)
  (empty shot139)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker175 l0)
  (shaker-level shaker175 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail320 ingredient101)
  (cocktail-part2 cocktail320 ingredient474)
)
 (:goal
  (and
      (contains shot456 cocktail320)
      (contains shot54 cocktail320)
)))

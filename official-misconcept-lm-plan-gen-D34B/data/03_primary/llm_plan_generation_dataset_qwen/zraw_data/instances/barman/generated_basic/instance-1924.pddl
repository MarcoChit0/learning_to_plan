(define (problem prob)
 (:domain barman)
 (:objects 
      shaker268 - shaker
      left right - hand
      shot361 - shot
      ingredient403 ingredient454 - ingredient
      cocktail429 - cocktail
      dispenser80 dispenser104 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker268)
  (ontable shot361)
  (dispenses dispenser80 ingredient403)
  (dispenses dispenser104 ingredient454)
  (clean shaker268)
  (clean shot361)
  (empty shaker268)
  (empty shot361)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker268 l0)
  (shaker-level shaker268 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail429 ingredient454)
  (cocktail-part2 cocktail429 ingredient403)
)
 (:goal
  (and
      (contains shot361 cocktail429)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker420 - shaker
      left right - hand
      shot429 - shot
      ingredient101 ingredient384 - ingredient
      cocktail1 - cocktail
      dispenser276 dispenser263 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker420)
  (ontable shot429)
  (dispenses dispenser276 ingredient101)
  (dispenses dispenser263 ingredient384)
  (clean shaker420)
  (clean shot429)
  (empty shaker420)
  (empty shot429)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker420 l0)
  (shaker-level shaker420 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient384)
  (cocktail-part2 cocktail1 ingredient101)
)
 (:goal
  (and
      (contains shot429 cocktail1)
)))

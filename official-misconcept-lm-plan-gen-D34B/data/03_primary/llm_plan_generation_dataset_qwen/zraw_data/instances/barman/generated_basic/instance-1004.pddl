(define (problem prob)
 (:domain barman)
 (:objects 
      shaker235 - shaker
      left right - hand
      shot34 shot256 - shot
      ingredient280 ingredient231 - ingredient
      cocktail1 - cocktail
      dispenser52 dispenser216 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker235)
  (ontable shot34)
  (ontable shot256)
  (dispenses dispenser52 ingredient280)
  (dispenses dispenser216 ingredient231)
  (clean shaker235)
  (clean shot34)
  (clean shot256)
  (empty shaker235)
  (empty shot34)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker235 l0)
  (shaker-level shaker235 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient280)
  (cocktail-part2 cocktail1 ingredient231)
)
 (:goal
  (and
      (contains shot34 cocktail1)
)))

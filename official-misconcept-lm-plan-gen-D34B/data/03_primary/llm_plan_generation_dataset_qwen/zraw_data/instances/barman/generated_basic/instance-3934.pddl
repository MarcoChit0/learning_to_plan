(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot236 shot349 - shot
      ingredient473 ingredient143 - ingredient
      cocktail62 - cocktail
      dispenser197 dispenser346 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot236)
  (ontable shot349)
  (dispenses dispenser197 ingredient473)
  (dispenses dispenser346 ingredient143)
  (clean shaker222)
  (clean shot236)
  (clean shot349)
  (empty shaker222)
  (empty shot236)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail62 ingredient473)
  (cocktail-part2 cocktail62 ingredient143)
)
 (:goal
  (and
      (contains shot236 cocktail62)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker429 - shaker
      left right - hand
      shot126 shot192 - shot
      ingredient346 ingredient115 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser87 dispenser132 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker429)
  (ontable shot126)
  (ontable shot192)
  (dispenses dispenser87 ingredient346)
  (dispenses dispenser132 ingredient115)
  (clean shaker429)
  (clean shot126)
  (clean shot192)
  (empty shaker429)
  (empty shot126)
  (empty shot192)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker429 l0)
  (shaker-level shaker429 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient115)
  (cocktail-part2 cocktail1 ingredient346)
  (cocktail-part1 cocktail2 ingredient115)
  (cocktail-part2 cocktail2 ingredient346)
)
 (:goal
  (and
      (contains shot126 cocktail2)
      (contains shot192 cocktail1)
)))

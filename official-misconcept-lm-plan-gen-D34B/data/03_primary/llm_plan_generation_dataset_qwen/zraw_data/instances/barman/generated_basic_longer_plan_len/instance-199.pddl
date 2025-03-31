(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot373 shot391 shot300 - shot
      ingredient103 ingredient61 - ingredient
      cocktail1 - cocktail
      dispenser462 dispenser9 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot373)
  (ontable shot391)
  (ontable shot300)
  (dispenses dispenser462 ingredient103)
  (dispenses dispenser9 ingredient61)
  (clean shaker276)
  (clean shot373)
  (clean shot391)
  (clean shot300)
  (empty shaker276)
  (empty shot373)
  (empty shot391)
  (empty shot300)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient61)
  (cocktail-part2 cocktail1 ingredient103)
)
 (:goal
  (and
      (contains shot373 cocktail1)
      (contains shot391 ingredient103)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker477 - shaker
      left right - hand
      shot34 shot393 - shot
      ingredient148 ingredient272 ingredient487 - ingredient
      cocktail1 - cocktail
      dispenser156 dispenser255 dispenser26 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker477)
  (ontable shot34)
  (ontable shot393)
  (dispenses dispenser156 ingredient148)
  (dispenses dispenser255 ingredient272)
  (dispenses dispenser26 ingredient487)
  (clean shaker477)
  (clean shot34)
  (clean shot393)
  (empty shaker477)
  (empty shot34)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker477 l0)
  (shaker-level shaker477 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient148)
  (cocktail-part2 cocktail1 ingredient272)
)
 (:goal
  (and
      (contains shot34 cocktail1)
)))

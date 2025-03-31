(define (problem prob)
 (:domain barman)
 (:objects 
      shaker53 - shaker
      left right - hand
      shot148 shot373 - shot
      ingredient95 ingredient6 ingredient119 ingredient191 - ingredient
      cocktail1 - cocktail
      dispenser67 dispenser351 dispenser462 dispenser271 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker53)
  (ontable shot148)
  (ontable shot373)
  (dispenses dispenser67 ingredient95)
  (dispenses dispenser351 ingredient6)
  (dispenses dispenser462 ingredient119)
  (dispenses dispenser271 ingredient191)
  (clean shaker53)
  (clean shot148)
  (clean shot373)
  (empty shaker53)
  (empty shot148)
  (empty shot373)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker53 l0)
  (shaker-level shaker53 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient119)
  (cocktail-part2 cocktail1 ingredient6)
)
 (:goal
  (and
      (contains shot148 cocktail1)
)))

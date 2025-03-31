(define (problem prob)
 (:domain barman)
 (:objects 
      shaker26 - shaker
      left right - hand
      shot47 shot259 - shot
      ingredient490 ingredient175 - ingredient
      cocktail174 - cocktail
      dispenser40 dispenser376 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker26)
  (ontable shot47)
  (ontable shot259)
  (dispenses dispenser40 ingredient490)
  (dispenses dispenser376 ingredient175)
  (clean shaker26)
  (clean shot47)
  (clean shot259)
  (empty shaker26)
  (empty shot47)
  (empty shot259)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker26 l0)
  (shaker-level shaker26 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient175)
  (cocktail-part2 cocktail174 ingredient490)
)
 (:goal
  (and
      (contains shot47 cocktail174)
)))

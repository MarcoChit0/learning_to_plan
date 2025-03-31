(define (problem prob)
 (:domain barman)
 (:objects 
      shaker146 - shaker
      left right - hand
      shot339 shot350 - shot
      ingredient409 ingredient270 - ingredient
      cocktail174 - cocktail
      dispenser226 dispenser464 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker146)
  (ontable shot339)
  (ontable shot350)
  (dispenses dispenser226 ingredient409)
  (dispenses dispenser464 ingredient270)
  (clean shaker146)
  (clean shot339)
  (clean shot350)
  (empty shaker146)
  (empty shot339)
  (empty shot350)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker146 l0)
  (shaker-level shaker146 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail174 ingredient270)
  (cocktail-part2 cocktail174 ingredient409)
)
 (:goal
  (and
      (contains shot339 cocktail174)
)))

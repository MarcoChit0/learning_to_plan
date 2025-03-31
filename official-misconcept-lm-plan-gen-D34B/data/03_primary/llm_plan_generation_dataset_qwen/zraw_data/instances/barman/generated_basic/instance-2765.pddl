(define (problem prob)
 (:domain barman)
 (:objects 
      shaker162 - shaker
      left right - hand
      shot64 shot34 - shot
      ingredient75 ingredient334 ingredient164 - ingredient
      cocktail1 - cocktail
      dispenser45 dispenser83 dispenser128 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker162)
  (ontable shot64)
  (ontable shot34)
  (dispenses dispenser45 ingredient75)
  (dispenses dispenser83 ingredient334)
  (dispenses dispenser128 ingredient164)
  (clean shaker162)
  (clean shot64)
  (clean shot34)
  (empty shaker162)
  (empty shot64)
  (empty shot34)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker162 l0)
  (shaker-level shaker162 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient75)
  (cocktail-part2 cocktail1 ingredient164)
)
 (:goal
  (and
      (contains shot64 cocktail1)
)))

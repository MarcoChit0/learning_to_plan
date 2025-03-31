(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot33 shot281 shot272 - shot
      ingredient192 ingredient268 - ingredient
      cocktail44 - cocktail
      dispenser318 dispenser49 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot33)
  (ontable shot281)
  (ontable shot272)
  (dispenses dispenser318 ingredient192)
  (dispenses dispenser49 ingredient268)
  (clean shaker191)
  (clean shot33)
  (clean shot281)
  (clean shot272)
  (empty shaker191)
  (empty shot33)
  (empty shot281)
  (empty shot272)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail44 ingredient192)
  (cocktail-part2 cocktail44 ingredient268)
)
 (:goal
  (and
      (contains shot33 cocktail44)
      (contains shot281 cocktail44)
)))

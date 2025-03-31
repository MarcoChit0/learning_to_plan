(define (problem prob)
 (:domain barman)
 (:objects 
      shaker250 - shaker
      left right - hand
      shot369 shot270 - shot
      ingredient225 ingredient32 ingredient316 ingredient116 - ingredient
      cocktail470 - cocktail
      dispenser218 dispenser225 dispenser99 dispenser101 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker250)
  (ontable shot369)
  (ontable shot270)
  (dispenses dispenser218 ingredient225)
  (dispenses dispenser225 ingredient32)
  (dispenses dispenser99 ingredient316)
  (dispenses dispenser101 ingredient116)
  (clean shaker250)
  (clean shot369)
  (clean shot270)
  (empty shaker250)
  (empty shot369)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker250 l0)
  (shaker-level shaker250 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail470 ingredient116)
  (cocktail-part2 cocktail470 ingredient32)
)
 (:goal
  (and
      (contains shot369 cocktail470)
)))

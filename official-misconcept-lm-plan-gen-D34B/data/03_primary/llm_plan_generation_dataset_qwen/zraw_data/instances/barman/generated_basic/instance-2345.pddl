(define (problem prob)
 (:domain barman)
 (:objects 
      shaker267 - shaker
      left right - hand
      shot422 shot321 - shot
      ingredient410 ingredient32 ingredient400 ingredient475 - ingredient
      cocktail323 - cocktail
      dispenser1 dispenser345 dispenser270 dispenser127 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker267)
  (ontable shot422)
  (ontable shot321)
  (dispenses dispenser1 ingredient410)
  (dispenses dispenser345 ingredient32)
  (dispenses dispenser270 ingredient400)
  (dispenses dispenser127 ingredient475)
  (clean shaker267)
  (clean shot422)
  (clean shot321)
  (empty shaker267)
  (empty shot422)
  (empty shot321)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker267 l0)
  (shaker-level shaker267 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail323 ingredient400)
  (cocktail-part2 cocktail323 ingredient32)
)
 (:goal
  (and
      (contains shot422 cocktail323)
)))

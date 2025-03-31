(define (problem prob)
 (:domain barman)
 (:objects 
      shaker173 - shaker
      left right - hand
      shot335 shot349 - shot
      ingredient347 ingredient268 - ingredient
      cocktail186 - cocktail
      dispenser187 dispenser225 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker173)
  (ontable shot335)
  (ontable shot349)
  (dispenses dispenser187 ingredient347)
  (dispenses dispenser225 ingredient268)
  (clean shaker173)
  (clean shot335)
  (clean shot349)
  (empty shaker173)
  (empty shot335)
  (empty shot349)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker173 l0)
  (shaker-level shaker173 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail186 ingredient268)
  (cocktail-part2 cocktail186 ingredient347)
)
 (:goal
  (and
      (contains shot335 cocktail186)
)))

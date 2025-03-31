(define (problem prob)
 (:domain barman)
 (:objects 
      shaker460 - shaker
      left right - hand
      shot287 - shot
      ingredient409 ingredient143 - ingredient
      cocktail1 - cocktail
      dispenser243 dispenser147 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker460)
  (ontable shot287)
  (dispenses dispenser243 ingredient409)
  (dispenses dispenser147 ingredient143)
  (clean shaker460)
  (clean shot287)
  (empty shaker460)
  (empty shot287)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker460 l0)
  (shaker-level shaker460 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient143)
  (cocktail-part2 cocktail1 ingredient409)
)
 (:goal
  (and
      (contains shot287 cocktail1)
)))

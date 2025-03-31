(define (problem prob)
 (:domain barman)
 (:objects 
      shaker400 - shaker
      left right - hand
      shot64 shot245 - shot
      ingredient249 ingredient313 - ingredient
      cocktail1 cocktail2 - cocktail
      dispenser50 dispenser1 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker400)
  (ontable shot64)
  (ontable shot245)
  (dispenses dispenser50 ingredient249)
  (dispenses dispenser1 ingredient313)
  (clean shaker400)
  (clean shot64)
  (clean shot245)
  (empty shaker400)
  (empty shot64)
  (empty shot245)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker400 l0)
  (shaker-level shaker400 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient249)
  (cocktail-part2 cocktail1 ingredient313)
  (cocktail-part1 cocktail2 ingredient249)
  (cocktail-part2 cocktail2 ingredient313)
)
 (:goal
  (and
      (contains shot64 cocktail2)
      (contains shot245 cocktail1)
)))

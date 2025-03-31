(define (problem prob)
 (:domain barman)
 (:objects 
      shaker336 - shaker
      left right - hand
      shot198 shot169 - shot
      ingredient367 ingredient352 ingredient416 ingredient301 - ingredient
      cocktail1 - cocktail
      dispenser491 dispenser96 dispenser360 dispenser161 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker336)
  (ontable shot198)
  (ontable shot169)
  (dispenses dispenser491 ingredient367)
  (dispenses dispenser96 ingredient352)
  (dispenses dispenser360 ingredient416)
  (dispenses dispenser161 ingredient301)
  (clean shaker336)
  (clean shot198)
  (clean shot169)
  (empty shaker336)
  (empty shot198)
  (empty shot169)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker336 l0)
  (shaker-level shaker336 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient301)
  (cocktail-part2 cocktail1 ingredient352)
)
 (:goal
  (and
      (contains shot198 cocktail1)
)))

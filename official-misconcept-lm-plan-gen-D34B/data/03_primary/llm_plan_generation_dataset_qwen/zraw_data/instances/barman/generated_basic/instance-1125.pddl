(define (problem prob)
 (:domain barman)
 (:objects 
      shaker1 - shaker
      left right - hand
      shot250 shot0 - shot
      ingredient280 ingredient201 ingredient271 - ingredient
      cocktail1 - cocktail
      dispenser474 dispenser354 dispenser411 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker1)
  (ontable shot250)
  (ontable shot0)
  (dispenses dispenser474 ingredient280)
  (dispenses dispenser354 ingredient201)
  (dispenses dispenser411 ingredient271)
  (clean shaker1)
  (clean shot250)
  (clean shot0)
  (empty shaker1)
  (empty shot250)
  (empty shot0)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker1 l0)
  (shaker-level shaker1 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient201)
  (cocktail-part2 cocktail1 ingredient271)
)
 (:goal
  (and
      (contains shot250 cocktail1)
)))

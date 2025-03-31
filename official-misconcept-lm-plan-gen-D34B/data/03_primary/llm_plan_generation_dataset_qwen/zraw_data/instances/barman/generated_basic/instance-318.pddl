(define (problem prob)
 (:domain barman)
 (:objects 
      shaker276 - shaker
      left right - hand
      shot25 shot39 - shot
      ingredient340 ingredient301 - ingredient
      cocktail1 - cocktail
      dispenser443 dispenser45 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker276)
  (ontable shot25)
  (ontable shot39)
  (dispenses dispenser443 ingredient340)
  (dispenses dispenser45 ingredient301)
  (clean shaker276)
  (clean shot25)
  (clean shot39)
  (empty shaker276)
  (empty shot25)
  (empty shot39)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker276 l0)
  (shaker-level shaker276 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient340)
  (cocktail-part2 cocktail1 ingredient301)
)
 (:goal
  (and
      (contains shot25 cocktail1)
)))

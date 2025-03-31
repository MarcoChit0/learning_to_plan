(define (problem prob)
 (:domain barman)
 (:objects 
      shaker431 - shaker
      left right - hand
      shot301 shot293 - shot
      ingredient145 ingredient13 - ingredient
      cocktail115 - cocktail
      dispenser203 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker431)
  (ontable shot301)
  (ontable shot293)
  (dispenses dispenser203 ingredient145)
  (dispenses dispenser437 ingredient13)
  (clean shaker431)
  (clean shot301)
  (clean shot293)
  (empty shaker431)
  (empty shot301)
  (empty shot293)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker431 l0)
  (shaker-level shaker431 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient145)
  (cocktail-part2 cocktail115 ingredient13)
)
 (:goal
  (and
      (contains shot301 cocktail115)
)))

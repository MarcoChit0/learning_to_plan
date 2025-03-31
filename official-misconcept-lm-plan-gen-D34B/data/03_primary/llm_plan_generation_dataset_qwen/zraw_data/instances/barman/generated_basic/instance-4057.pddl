(define (problem prob)
 (:domain barman)
 (:objects 
      shaker199 - shaker
      left right - hand
      shot369 shot279 - shot
      ingredient319 ingredient70 - ingredient
      cocktail301 - cocktail
      dispenser260 dispenser437 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker199)
  (ontable shot369)
  (ontable shot279)
  (dispenses dispenser260 ingredient319)
  (dispenses dispenser437 ingredient70)
  (clean shaker199)
  (clean shot369)
  (clean shot279)
  (empty shaker199)
  (empty shot369)
  (empty shot279)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker199 l0)
  (shaker-level shaker199 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail301 ingredient319)
  (cocktail-part2 cocktail301 ingredient70)
)
 (:goal
  (and
      (contains shot369 cocktail301)
)))

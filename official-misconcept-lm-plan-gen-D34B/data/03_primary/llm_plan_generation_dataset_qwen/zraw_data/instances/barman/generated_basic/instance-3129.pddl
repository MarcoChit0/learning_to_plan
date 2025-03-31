(define (problem prob)
 (:domain barman)
 (:objects 
      shaker225 - shaker
      left right - hand
      shot81 shot307 shot360 - shot
      ingredient205 ingredient86 - ingredient
      cocktail122 - cocktail
      dispenser362 dispenser157 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker225)
  (ontable shot81)
  (ontable shot307)
  (ontable shot360)
  (dispenses dispenser362 ingredient205)
  (dispenses dispenser157 ingredient86)
  (clean shaker225)
  (clean shot81)
  (clean shot307)
  (clean shot360)
  (empty shaker225)
  (empty shot81)
  (empty shot307)
  (empty shot360)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker225 l0)
  (shaker-level shaker225 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail122 ingredient86)
  (cocktail-part2 cocktail122 ingredient205)
)
 (:goal
  (and
      (contains shot81 cocktail122)
      (contains shot307 cocktail122)
)))

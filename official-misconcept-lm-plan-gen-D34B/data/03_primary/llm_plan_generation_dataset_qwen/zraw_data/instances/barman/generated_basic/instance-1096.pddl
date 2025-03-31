(define (problem prob)
 (:domain barman)
 (:objects 
      shaker301 - shaker
      left right - hand
      shot441 - shot
      ingredient464 ingredient467 ingredient415 - ingredient
      cocktail1 - cocktail
      dispenser353 dispenser165 dispenser444 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker301)
  (ontable shot441)
  (dispenses dispenser353 ingredient464)
  (dispenses dispenser165 ingredient467)
  (dispenses dispenser444 ingredient415)
  (clean shaker301)
  (clean shot441)
  (empty shaker301)
  (empty shot441)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker301 l0)
  (shaker-level shaker301 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient415)
  (cocktail-part2 cocktail1 ingredient464)
)
 (:goal
  (and
      (contains shot441 cocktail1)
)))

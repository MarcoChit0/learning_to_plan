(define (problem prob)
 (:domain barman)
 (:objects 
      shaker335 - shaker
      left right - hand
      shot58 shot301 - shot
      ingredient331 ingredient177 ingredient163 ingredient174 - ingredient
      cocktail1 - cocktail
      dispenser387 dispenser434 dispenser114 dispenser146 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker335)
  (ontable shot58)
  (ontable shot301)
  (dispenses dispenser387 ingredient331)
  (dispenses dispenser434 ingredient177)
  (dispenses dispenser114 ingredient163)
  (dispenses dispenser146 ingredient174)
  (clean shaker335)
  (clean shot58)
  (clean shot301)
  (empty shaker335)
  (empty shot58)
  (empty shot301)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker335 l0)
  (shaker-level shaker335 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient177)
  (cocktail-part2 cocktail1 ingredient174)
)
 (:goal
  (and
      (contains shot58 cocktail1)
)))

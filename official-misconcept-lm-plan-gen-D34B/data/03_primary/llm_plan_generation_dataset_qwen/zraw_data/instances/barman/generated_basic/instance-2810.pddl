(define (problem prob)
 (:domain barman)
 (:objects 
      shaker378 - shaker
      left right - hand
      shot422 shot199 - shot
      ingredient265 ingredient93 - ingredient
      cocktail461 - cocktail
      dispenser37 dispenser61 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker378)
  (ontable shot422)
  (ontable shot199)
  (dispenses dispenser37 ingredient265)
  (dispenses dispenser61 ingredient93)
  (clean shaker378)
  (clean shot422)
  (clean shot199)
  (empty shaker378)
  (empty shot422)
  (empty shot199)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker378 l0)
  (shaker-level shaker378 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail461 ingredient93)
  (cocktail-part2 cocktail461 ingredient265)
)
 (:goal
  (and
      (contains shot422 cocktail461)
)))

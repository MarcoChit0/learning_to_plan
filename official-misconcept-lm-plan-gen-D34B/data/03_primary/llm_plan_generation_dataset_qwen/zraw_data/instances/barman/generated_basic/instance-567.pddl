(define (problem prob)
 (:domain barman)
 (:objects 
      shaker387 - shaker
      left right - hand
      shot353 shot177 - shot
      ingredient177 ingredient317 ingredient217 - ingredient
      cocktail1 - cocktail
      dispenser462 dispenser260 dispenser385 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker387)
  (ontable shot353)
  (ontable shot177)
  (dispenses dispenser462 ingredient177)
  (dispenses dispenser260 ingredient317)
  (dispenses dispenser385 ingredient217)
  (clean shaker387)
  (clean shot353)
  (clean shot177)
  (empty shaker387)
  (empty shot353)
  (empty shot177)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker387 l0)
  (shaker-level shaker387 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient317)
  (cocktail-part2 cocktail1 ingredient177)
)
 (:goal
  (and
      (contains shot353 cocktail1)
)))

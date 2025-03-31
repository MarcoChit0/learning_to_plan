(define (problem prob)
 (:domain barman)
 (:objects 
      shaker467 - shaker
      left right - hand
      shot208 shot175 shot146 - shot
      ingredient186 ingredient437 ingredient344 ingredient360 - ingredient
      cocktail141 - cocktail
      dispenser212 dispenser274 dispenser67 dispenser429 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker467)
  (ontable shot208)
  (ontable shot175)
  (ontable shot146)
  (dispenses dispenser212 ingredient186)
  (dispenses dispenser274 ingredient437)
  (dispenses dispenser67 ingredient344)
  (dispenses dispenser429 ingredient360)
  (clean shaker467)
  (clean shot208)
  (clean shot175)
  (clean shot146)
  (empty shaker467)
  (empty shot208)
  (empty shot175)
  (empty shot146)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker467 l0)
  (shaker-level shaker467 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail141 ingredient186)
  (cocktail-part2 cocktail141 ingredient344)
)
 (:goal
  (and
      (contains shot208 cocktail141)
      (contains shot175 ingredient186)
)))

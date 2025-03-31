(define (problem prob)
 (:domain barman)
 (:objects 
      shaker178 - shaker
      left right - hand
      shot123 shot309 - shot
      ingredient456 ingredient126 ingredient22 - ingredient
      cocktail358 - cocktail
      dispenser98 dispenser463 dispenser196 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker178)
  (ontable shot123)
  (ontable shot309)
  (dispenses dispenser98 ingredient456)
  (dispenses dispenser463 ingredient126)
  (dispenses dispenser196 ingredient22)
  (clean shaker178)
  (clean shot123)
  (clean shot309)
  (empty shaker178)
  (empty shot123)
  (empty shot309)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker178 l0)
  (shaker-level shaker178 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail358 ingredient22)
  (cocktail-part2 cocktail358 ingredient456)
)
 (:goal
  (and
      (contains shot123 cocktail358)
)))

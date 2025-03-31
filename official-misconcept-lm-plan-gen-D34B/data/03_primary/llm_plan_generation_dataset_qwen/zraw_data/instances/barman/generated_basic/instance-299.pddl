(define (problem prob)
 (:domain barman)
 (:objects 
      shaker222 - shaker
      left right - hand
      shot142 shot145 - shot
      ingredient219 ingredient480 - ingredient
      cocktail1 - cocktail
      dispenser212 dispenser89 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker222)
  (ontable shot142)
  (ontable shot145)
  (dispenses dispenser212 ingredient219)
  (dispenses dispenser89 ingredient480)
  (clean shaker222)
  (clean shot142)
  (clean shot145)
  (empty shaker222)
  (empty shot142)
  (empty shot145)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker222 l0)
  (shaker-level shaker222 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient480)
  (cocktail-part2 cocktail1 ingredient219)
)
 (:goal
  (and
      (contains shot142 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker183 - shaker
      left right - hand
      shot148 shot184 shot268 - shot
      ingredient462 ingredient26 ingredient313 ingredient248 - ingredient
      cocktail142 - cocktail
      dispenser249 dispenser181 dispenser91 dispenser333 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker183)
  (ontable shot148)
  (ontable shot184)
  (ontable shot268)
  (dispenses dispenser249 ingredient462)
  (dispenses dispenser181 ingredient26)
  (dispenses dispenser91 ingredient313)
  (dispenses dispenser333 ingredient248)
  (clean shaker183)
  (clean shot148)
  (clean shot184)
  (clean shot268)
  (empty shaker183)
  (empty shot148)
  (empty shot184)
  (empty shot268)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker183 l0)
  (shaker-level shaker183 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail142 ingredient462)
  (cocktail-part2 cocktail142 ingredient248)
)
 (:goal
  (and
      (contains shot148 cocktail142)
      (contains shot184 cocktail142)
)))

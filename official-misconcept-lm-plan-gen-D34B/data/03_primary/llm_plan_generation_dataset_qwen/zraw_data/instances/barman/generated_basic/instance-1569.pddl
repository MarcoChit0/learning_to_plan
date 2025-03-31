(define (problem prob)
 (:domain barman)
 (:objects 
      shaker164 - shaker
      left right - hand
      shot372 shot39 shot277 - shot
      ingredient173 ingredient134 ingredient189 ingredient83 - ingredient
      cocktail277 - cocktail
      dispenser63 dispenser338 dispenser477 dispenser387 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker164)
  (ontable shot372)
  (ontable shot39)
  (ontable shot277)
  (dispenses dispenser63 ingredient173)
  (dispenses dispenser338 ingredient134)
  (dispenses dispenser477 ingredient189)
  (dispenses dispenser387 ingredient83)
  (clean shaker164)
  (clean shot372)
  (clean shot39)
  (clean shot277)
  (empty shaker164)
  (empty shot372)
  (empty shot39)
  (empty shot277)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker164 l0)
  (shaker-level shaker164 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail277 ingredient189)
  (cocktail-part2 cocktail277 ingredient83)
)
 (:goal
  (and
      (contains shot372 cocktail277)
      (contains shot39 cocktail277)
)))

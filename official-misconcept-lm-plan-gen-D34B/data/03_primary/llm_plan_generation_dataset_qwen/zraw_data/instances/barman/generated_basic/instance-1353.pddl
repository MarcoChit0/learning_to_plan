(define (problem prob)
 (:domain barman)
 (:objects 
      shaker416 - shaker
      left right - hand
      shot392 shot175 shot270 - shot
      ingredient44 ingredient488 ingredient88 - ingredient
      cocktail184 - cocktail
      dispenser159 dispenser214 dispenser478 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker416)
  (ontable shot392)
  (ontable shot175)
  (ontable shot270)
  (dispenses dispenser159 ingredient44)
  (dispenses dispenser214 ingredient488)
  (dispenses dispenser478 ingredient88)
  (clean shaker416)
  (clean shot392)
  (clean shot175)
  (clean shot270)
  (empty shaker416)
  (empty shot392)
  (empty shot175)
  (empty shot270)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker416 l0)
  (shaker-level shaker416 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail184 ingredient44)
  (cocktail-part2 cocktail184 ingredient488)
)
 (:goal
  (and
      (contains shot392 cocktail184)
      (contains shot175 cocktail184)
)))

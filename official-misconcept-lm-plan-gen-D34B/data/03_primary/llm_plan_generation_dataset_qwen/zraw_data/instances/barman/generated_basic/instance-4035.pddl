(define (problem prob)
 (:domain barman)
 (:objects 
      shaker302 - shaker
      left right - hand
      shot392 shot232 - shot
      ingredient429 ingredient239 ingredient395 ingredient166 - ingredient
      cocktail251 - cocktail
      dispenser136 dispenser462 dispenser158 dispenser1 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker302)
  (ontable shot392)
  (ontable shot232)
  (dispenses dispenser136 ingredient429)
  (dispenses dispenser462 ingredient239)
  (dispenses dispenser158 ingredient395)
  (dispenses dispenser1 ingredient166)
  (clean shaker302)
  (clean shot392)
  (clean shot232)
  (empty shaker302)
  (empty shot392)
  (empty shot232)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker302 l0)
  (shaker-level shaker302 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail251 ingredient239)
  (cocktail-part2 cocktail251 ingredient429)
)
 (:goal
  (and
      (contains shot392 cocktail251)
)))

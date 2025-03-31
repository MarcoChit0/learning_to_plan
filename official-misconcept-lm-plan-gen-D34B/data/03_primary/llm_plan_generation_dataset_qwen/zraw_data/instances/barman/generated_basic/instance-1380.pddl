(define (problem prob)
 (:domain barman)
 (:objects 
      shaker180 - shaker
      left right - hand
      shot149 shot27 shot246 - shot
      ingredient496 ingredient332 ingredient83 ingredient318 - ingredient
      cocktail18 - cocktail
      dispenser150 dispenser144 dispenser256 dispenser226 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker180)
  (ontable shot149)
  (ontable shot27)
  (ontable shot246)
  (dispenses dispenser150 ingredient496)
  (dispenses dispenser144 ingredient332)
  (dispenses dispenser256 ingredient83)
  (dispenses dispenser226 ingredient318)
  (clean shaker180)
  (clean shot149)
  (clean shot27)
  (clean shot246)
  (empty shaker180)
  (empty shot149)
  (empty shot27)
  (empty shot246)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker180 l0)
  (shaker-level shaker180 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail18 ingredient83)
  (cocktail-part2 cocktail18 ingredient496)
)
 (:goal
  (and
      (contains shot149 cocktail18)
      (contains shot27 cocktail18)
)))

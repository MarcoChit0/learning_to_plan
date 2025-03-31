(define (problem prob)
 (:domain barman)
 (:objects 
      shaker108 - shaker
      left right - hand
      shot336 shot395 shot256 - shot
      ingredient181 ingredient113 ingredient18 ingredient393 - ingredient
      cocktail1 - cocktail
      dispenser155 dispenser335 dispenser88 dispenser405 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker108)
  (ontable shot336)
  (ontable shot395)
  (ontable shot256)
  (dispenses dispenser155 ingredient181)
  (dispenses dispenser335 ingredient113)
  (dispenses dispenser88 ingredient18)
  (dispenses dispenser405 ingredient393)
  (clean shaker108)
  (clean shot336)
  (clean shot395)
  (clean shot256)
  (empty shaker108)
  (empty shot336)
  (empty shot395)
  (empty shot256)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker108 l0)
  (shaker-level shaker108 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient181)
  (cocktail-part2 cocktail1 ingredient18)
)
 (:goal
  (and
      (contains shot336 cocktail1)
      (contains shot395 cocktail1)
)))

(define (problem prob)
 (:domain barman)
 (:objects 
      shaker310 - shaker
      left right - hand
      shot250 shot187 shot297 - shot
      ingredient187 ingredient59 - ingredient
      cocktail115 - cocktail
      dispenser45 dispenser404 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker310)
  (ontable shot250)
  (ontable shot187)
  (ontable shot297)
  (dispenses dispenser45 ingredient187)
  (dispenses dispenser404 ingredient59)
  (clean shaker310)
  (clean shot250)
  (clean shot187)
  (clean shot297)
  (empty shaker310)
  (empty shot250)
  (empty shot187)
  (empty shot297)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker310 l0)
  (shaker-level shaker310 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail115 ingredient187)
  (cocktail-part2 cocktail115 ingredient59)
)
 (:goal
  (and
      (contains shot250 cocktail115)
      (contains shot187 ingredient59)
)))

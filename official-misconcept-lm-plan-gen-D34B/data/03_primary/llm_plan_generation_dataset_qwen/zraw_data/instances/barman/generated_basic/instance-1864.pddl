(define (problem prob)
 (:domain barman)
 (:objects 
      shaker446 - shaker
      left right - hand
      shot449 shot283 - shot
      ingredient217 ingredient219 - ingredient
      cocktail378 - cocktail
      dispenser285 dispenser266 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker446)
  (ontable shot449)
  (ontable shot283)
  (dispenses dispenser285 ingredient217)
  (dispenses dispenser266 ingredient219)
  (clean shaker446)
  (clean shot449)
  (clean shot283)
  (empty shaker446)
  (empty shot449)
  (empty shot283)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker446 l0)
  (shaker-level shaker446 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail378 ingredient219)
  (cocktail-part2 cocktail378 ingredient217)
)
 (:goal
  (and
      (contains shot449 cocktail378)
)))

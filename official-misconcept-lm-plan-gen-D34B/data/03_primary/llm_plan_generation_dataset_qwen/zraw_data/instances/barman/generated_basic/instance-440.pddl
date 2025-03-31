(define (problem prob)
 (:domain barman)
 (:objects 
      shaker473 - shaker
      left right - hand
      shot392 shot401 - shot
      ingredient313 ingredient448 ingredient217 - ingredient
      cocktail1 - cocktail
      dispenser16 dispenser92 dispenser25 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker473)
  (ontable shot392)
  (ontable shot401)
  (dispenses dispenser16 ingredient313)
  (dispenses dispenser92 ingredient448)
  (dispenses dispenser25 ingredient217)
  (clean shaker473)
  (clean shot392)
  (clean shot401)
  (empty shaker473)
  (empty shot392)
  (empty shot401)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker473 l0)
  (shaker-level shaker473 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail1 ingredient448)
  (cocktail-part2 cocktail1 ingredient217)
)
 (:goal
  (and
      (contains shot392 cocktail1)
)))

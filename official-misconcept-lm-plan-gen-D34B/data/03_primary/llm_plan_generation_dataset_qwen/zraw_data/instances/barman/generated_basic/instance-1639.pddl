(define (problem prob)
 (:domain barman)
 (:objects 
      shaker191 - shaker
      left right - hand
      shot315 shot141 shot393 - shot
      ingredient201 ingredient200 - ingredient
      cocktail135 - cocktail
      dispenser33 dispenser251 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker191)
  (ontable shot315)
  (ontable shot141)
  (ontable shot393)
  (dispenses dispenser33 ingredient201)
  (dispenses dispenser251 ingredient200)
  (clean shaker191)
  (clean shot315)
  (clean shot141)
  (clean shot393)
  (empty shaker191)
  (empty shot315)
  (empty shot141)
  (empty shot393)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker191 l0)
  (shaker-level shaker191 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail135 ingredient200)
  (cocktail-part2 cocktail135 ingredient201)
)
 (:goal
  (and
      (contains shot315 cocktail135)
      (contains shot141 cocktail135)
)))

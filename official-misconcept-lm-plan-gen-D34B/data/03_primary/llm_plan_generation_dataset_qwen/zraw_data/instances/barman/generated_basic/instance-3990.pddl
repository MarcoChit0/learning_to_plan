(define (problem prob)
 (:domain barman)
 (:objects 
      shaker19 - shaker
      left right - hand
      shot435 shot434 - shot
      ingredient145 ingredient63 ingredient124 ingredient81 - ingredient
      cocktail407 - cocktail
      dispenser87 dispenser312 dispenser421 dispenser73 - dispenser
      l0 l1 l2 - level
)
 (:init 
  (ontable shaker19)
  (ontable shot435)
  (ontable shot434)
  (dispenses dispenser87 ingredient145)
  (dispenses dispenser312 ingredient63)
  (dispenses dispenser421 ingredient124)
  (dispenses dispenser73 ingredient81)
  (clean shaker19)
  (clean shot435)
  (clean shot434)
  (empty shaker19)
  (empty shot435)
  (empty shot434)
  (handempty left)
  (handempty right)
  (shaker-empty-level shaker19 l0)
  (shaker-level shaker19 l0)
  (next l0 l1)
  (next l1 l2)
  (cocktail-part1 cocktail407 ingredient63)
  (cocktail-part2 cocktail407 ingredient81)
)
 (:goal
  (and
      (contains shot435 cocktail407)
)))
